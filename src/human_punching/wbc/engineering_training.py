"""Measured canonical engineering learning around inherited WBC/RSL-RL PPO.

This entry never qualifies the reference domain, imports prior specialist/guard/
capacity weights, submits jobs, or promotes a policy. New lineages start from
scratch. Only its own verified full boundary checkpoints may continue learning,
with fresh physical trajectories and unchanged scientific/runtime contracts.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

from .budget import TrainingScale, plateau
from .conditioning import Conditioning
from .engineering_admission import SCOPE, _evidence, read_artifact
from .measurement_contract import build_measurement_contract, digest, verify_measurement_contract
from .provenance import PROJECT_ROOT, runtime_record, sha256_file, writable_path, write_json
from .validation import iteration_chunks
from .runtime.engineering_recipe import (LEGACY_RECIPE_ID, RECIPE_IDS, resolve_engineering_recipe,
                                         verify_recipe_manifest)

SCHEMA = "human_wbc_engineering_training_plan_v1"
MAX_ITERATIONS = 2000
INTERVAL = 250


def sampler_identity(seed: int, engineering_recipe=LEGACY_RECIPE_ID) -> dict:
    recipe = resolve_engineering_recipe(engineering_recipe)
    settings = recipe.training
    return {"scope": SCOPE, "origin": "new_scratch_shared_canonical_cycle",
        "training_seed": seed, "families": "uniform_four_families",
        "variants": "uniform_within_family",
        "initialization": "frame_zero_reference_and_strict_guard_mixture"
            if settings.reference_start_probability else "guard_start",
        "reference_state_initialization": bool(settings.reference_start_probability),
        "interior_reference_state_initialization": False,
        "reference_start_probability": settings.reference_start_probability,
        "strict_guard_start_probability": 1.-settings.reference_start_probability,
        "initial_support_grace_ticks": settings.initial_support_grace_ticks,
        "engineering_recipe": recipe.manifest(), "engineering_recipe_sha256": recipe.sha256,
        "training_starts_count_as_strict_ready_launches": False, "assistance": False,
        "capacity_or_historical_weights_allowed": False,
        "continuation": "full_learner_with_fresh_physical_trajectories_not_exact_simulator_continuation"}


def training_curriculum(engineering_recipe=LEGACY_RECIPE_ID) -> dict:
    recipe = resolve_engineering_recipe(engineering_recipe)
    settings = recipe.training
    return {"engineering_recipe": recipe.manifest(), "engineering_recipe_sha256": recipe.sha256,
        "strict_guard_start_probability": 1.-settings.reference_start_probability,
        "reference_start_probability": settings.reference_start_probability,
        "reference_start_scope": "frame_zero_at_ordinary_episode_reset_only",
        "initial_support_grace_ticks": settings.initial_support_grace_ticks,
        "guard_joint_posture_metric": settings.guard_joint_posture_metric,
        "guard_joint_posture_weight": settings.guard_joint_posture_weight,
        "interior_rsi": False, "assistance": False, "mid_cycle_state_replacement": False,
        "strict_validation_recipe": LEGACY_RECIPE_ID}


def _recipe_request(engineering_recipe, training_initialization_report):
    recipe = resolve_engineering_recipe(engineering_recipe)
    enabled = recipe.training.reference_start_probability > 0
    if enabled != (training_initialization_report is not None):
        raise ValueError("Revised engineering recipe requires native training initialization evidence; "
                         "legacy recipe must not supply it")
    return recipe


def _plan_recipe(plan):
    """Fail closed before constructing a worker from a mismatched prepared plan."""
    recipe = verify_recipe_manifest(plan.get("engineering_recipe"), plan.get("engineering_recipe_sha256"))
    identity = plan["measurement_contract"]["payload"]["reference_and_body"]
    if (identity.get("engineering_recipe") != recipe.manifest()
            or identity.get("engineering_recipe_sha256") != recipe.sha256
            or plan.get("sampler_identity") != sampler_identity(plan["seed"], recipe)):
        raise ValueError("Plan, sampler and measured engineering recipes differ")
    evidence = plan["inputs"].get("training_initialization_report")
    _recipe_request(recipe, evidence)
    return recipe


def _bounds(iterations, seed, completed=0):
    if (type(iterations) is not int or type(seed) is not int or type(completed) is not int
            or seed < 0 or completed < 0 or iterations < 1 or completed+iterations > MAX_ITERATIONS):
        raise ValueError("Canonical engineering requires a nonnegative seed and cumulative1..2000 iterations")


def _add_file(files: dict, path: Path, expected=None):
    record = _evidence(path)
    if expected is not None and record["sha256"] != expected:
        raise ValueError("Prepared input hash differs: "+record["path"])
    old = files.setdefault(record["path"], record["sha256"])
    if old != record["sha256"]:
        raise ValueError("Conflicting input identities")
    return record


def _collect_verified_records(files: dict, value):
    """Walk verified evidence structures, not historical packet source maps."""
    if isinstance(value, dict):
        if set(("path", "sha256")) <= value.keys():
            _add_file(files, read_artifact(value), value["sha256"])
        for name, child in value.items():
            if name in ("human_files", "source_hashes", "verified_physical_reference_files"):
                for relative, expected in child.items():
                    _add_file(files, PROJECT_ROOT/relative, expected)
            else:
                _collect_verified_records(files, child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _collect_verified_records(files, child)


def _resume_plan(path, manifest_hash, *, seed, reference_hash, source_hashes,
                 engineering_recipe=LEGACY_RECIPE_ID):
    from .checkpointing import CheckpointContract, assert_checkpoint_rank_parity, CONTINUATION
    if path is None:
        if manifest_hash is not None:
            raise ValueError("Resume manifest hash requires a checkpoint directory")
        return None, 0, {}
    root = read_artifact(_evidence(Path(path)/"manifest.json")).parent
    record = _evidence(root/"manifest.json")
    if record["sha256"] != manifest_hash:
        raise ValueError("Explicit full engineering checkpoint manifest hash required")
    manifest = json.loads((root/"manifest.json").read_text())
    contract = CheckpointContract(**manifest.get("contract", {}))
    progress = manifest.get("progress", {})
    validation = progress.get("validation", {})
    if (manifest.get("schema") != "human_wbc_boundary_checkpoint_v1"
            or manifest.get("status") != "complete" or manifest.get("world_size") != 2
            or manifest.get("contract_sha256") != contract.sha256
            or manifest.get("sampler_identity") != sampler_identity(seed, engineering_recipe)
            or progress.get("curriculum") != training_curriculum(engineering_recipe)
            or manifest.get("continuation") != CONTINUATION or manifest.get("physical_state_saved") is not False
            or progress.get("stage") != "canonical"
            or progress.get("completed_iterations") != progress.get("stage_completed_iterations")
            or validation.get("scope") != SCOPE or validation.get("resume_allowed") is not True
            or contract.reference_manifest_sha256 != reference_hash
            or contract.implementation_sources_sha256 != digest(source_hashes)):
        raise ValueError("Only this unchanged engineering lineage's full learner checkpoint can resume")
    receipts = manifest.get("ranks", [])
    assert_checkpoint_rank_parity(receipts)
    if len(receipts) != 2 or {r["rank"] for r in receipts} != {0, 1}:
        raise ValueError("Resume requires both complete rank states")
    files = {record["path"]: record["sha256"]}
    for row in receipts:
        member = (root/row["path"]).resolve(strict=True)
        if not member.is_relative_to(root):
            raise ValueError("Checkpoint member escapes its root")
        _add_file(files, member, row["sha256"])
    _collect_verified_records(files, validation.get("panels", []))
    return {"path": str(root.relative_to(PROJECT_ROOT)), "manifest_sha256": manifest_hash,
            "progress": progress}, progress["completed_iterations"], files


def checked_engineering_plan(bundle: Path, admission_path: Path, observer_report: Path,
                             bridge_report: Path, capacity_report: Path, *, iterations: int,
                             seed: int, resume: Path | None = None,
                             resume_manifest_sha256: str | None = None,
                             engineering_recipe=LEGACY_RECIPE_ID,
                             training_initialization_report: Path | None = None) -> dict:
    """CPU preparation gate; all paths/hashes are inspectable and deterministic."""
    from .capacity import CapacityConfig, aggregate_capacity
    from .capacity_evidence import load_engineering_admission
    from .runtime.bundle import load_runtime_bundle
    from .engineering_cycle_panel import build_engineering_recipes, engineering_recipe_bytes
    _bounds(iterations, seed)
    recipe = _recipe_request(engineering_recipe, training_initialization_report)
    checked = load_runtime_bundle(bundle)
    admission = load_engineering_admission(admission_path)
    scale = TrainingScale(global_environments=8192, world_size=2)
    contract = build_measurement_contract(checked, scale=scale, conditioning=Conditioning.COMBINED.value,
        engineering_admission=admission, observer_report=observer_report, bridge_report=bridge_report,
        engineering_recipe=recipe.recipe_id, training_initialization_report=training_initialization_report)
    files = {}
    records = {name: _add_file(files, path) for name, path in (
        ("engineering_admission", admission_path), ("observer_report", observer_report),
        ("bridge_report", bridge_report), ("capacity_report", capacity_report))}
    if training_initialization_report is not None:
        records["training_initialization_report"] = _add_file(files, training_initialization_report)
    capacity_path = read_artifact(records["capacity_report"])
    measured = json.loads(capacity_path.read_text())
    launch_path, execution_path, packet_path = (capacity_path.parent/"launch.json",
        capacity_path.parent.parent/"execution.json", capacity_path.parent.parent/"packet.json")
    launch, execution, packet = (json.loads(path.read_text()) for path in (launch_path, execution_path, packet_path))
    for path in (launch_path, execution_path, packet_path):
        _add_file(files, path)
    if (execution.get("schema") != "human_wbc_capacity_execution_v1" or execution.get("returncode") != 0
            or execution.get("budget_stop") is not None or not execution.get("slurm_job_id")
            or execution.get("packet_sha256") != sha256_file(packet_path)
            or packet.get("schema") != "human_wbc_engineering_capacity_packet_v1"):
        raise ValueError("Completed allocated native capacity evidence is required")
    config = CapacityConfig(**launch["config"])
    if (config.scale != scale or config.engineering_recipe != recipe.recipe_id
            or measured.get("learning_scope") != SCOPE
            or measured.get("schema") != "human_wbc_capacity_measurement_v1"
            or measured.get("status") != "passed" or measured.get("production_reference_qualified") is not False
            or measured.get("measurement_contract_sha256") != contract["sha256"]):
        raise ValueError("First engineering session requires matching8192-global/two-L40S native capacity")
    verify_measurement_contract(measured["measurement_contract"], contract)
    ranks = []
    for rank in range(2):
        path = capacity_path.parent/f"rank_{rank:02d}"/"measurement.json"
        _add_file(files, path)
        ranks.append(json.loads(path.read_text()))
    recomputed = aggregate_capacity(config, ranks, sha256_file(checked.root/"manifest.json"), contract)
    if any(measured.get(key) != value for key, value in recomputed.items()):
        raise ValueError("Native capacity aggregate differs from its rank evidence")
    if recomputed["status"] != "passed" or "L40S" not in recomputed["gpu_hardware"]["name"]:
        raise ValueError("Measured native PPO/memory/hardware gates did not pass")
    # The panel receipt is additional to PPO throughput. Its actor is disposable.
    from .capacity_panel import verify_post_measurement_panel
    panel = measured.get("post_measurement_panel", {})
    verified_panel = verify_post_measurement_panel(panel)
    expectation = panel.get("expectation", {})
    panel_contract = json.loads(read_artifact(panel["checkpoint_contract"]).read_text())
    if (expectation.get("measurement_contract_sha256") != contract["sha256"]
            or expectation.get("engineering_admission_sha256") != admission.sha256
            or expectation.get("reference_manifest_sha256") != sha256_file(checked.root/"manifest.json")
            or expectation.get("completed_iterations") != config.warmup_iterations+config.measurement_iterations
            or expectation.get("training_seed") != config.seed
            or panel_contract.get("sources") != contract["payload"]["sources"]
            or panel_contract["checkpoint"]["sha256"] != ranks[0].get("final_checkpoint_sha256")):
        raise ValueError("Frozen panel is not from this exact measured capacity learner/configuration")
    if (panel.get("status") != "passed" or panel.get("training_state_unchanged") is not True
            or not isinstance(panel.get("wall_time_s"), (int, float))
            or not math.isfinite(panel["wall_time_s"]) or panel["wall_time_s"] <= 0
            or not isinstance(panel.get("headroom_fraction"), (int, float))
            or not .15 <= panel["headroom_fraction"] <= 1.):
        raise ValueError("Native frozen-actor panel and training-memory coexistence must be measured first")
    _collect_verified_records(files, verified_panel)
    _collect_verified_records(files, panel)
    _collect_verified_records(files, admission.as_dict())
    _collect_verified_records(files, contract["payload"])
    for path in checked.root.rglob("*"):
        if path.is_file():
            _add_file(files, path)
    sources = contract["payload"]["sources"]["human_files"]
    for project, metadata in contract["payload"]["sources"]["upstream"].items():
        _add_file(files, PROJECT_ROOT/".cache/deps"/f"{project}-{metadata['revision'][:8]}"/".task10-source.json",
                  metadata["source_export_manifest_sha256"])
    resumed, completed, resume_files = _resume_plan(resume, resume_manifest_sha256, seed=seed,
        reference_hash=sha256_file(checked.root/"manifest.json"), source_hashes=sources,
        engineering_recipe=recipe.recipe_id)
    files.update(resume_files)
    _bounds(iterations, seed, completed)
    recipes = build_engineering_recipes(checked, episodes_per_family=16)
    recipe_bytes = engineering_recipe_bytes(recipes)
    chunks = iteration_chunks(completed, iterations)
    boundaries, total = [], completed
    for count in chunks:
        total += count; boundaries.append(total)
    projection = {"learning_seconds": iterations*scale.fresh_transitions/measured["transitions_per_second"],
        "measured_initialization_seconds": max(row["initialization_wall_time_s"] for row in ranks),
        "measured_panel_seconds": panel["wall_time_s"], "planned_panels": len(chunks),
        "panel_seconds": len(chunks)*panel["wall_time_s"],
        "qualification": "Measured scenarios, not a wall-time guarantee; packet adds shutdown/variability reserve."}
    payload = {"schema": SCHEMA, "scope": SCOPE, "production_qualified": False,
        "stage": "canonical", "seed": seed, "iterations": iterations,
        "prior_completed_iterations": completed, "planned_completed_iterations": total,
        "scale": asdict(scale), "conditioning": Conditioning.COMBINED.value,
        "bundle": str(checked.root.relative_to(PROJECT_ROOT)), "inputs": records,
        "measurement_contract": contract, "measurement_contract_sha256": contract["sha256"],
        "gpu_hardware": measured["gpu_hardware"], "engineering_admission_sha256": admission.sha256,
        "reference_manifest_sha256": sha256_file(checked.root/"manifest.json"),
        "resume": resumed, "sampler_identity": sampler_identity(seed, recipe),
        "engineering_recipe": recipe.manifest(), "engineering_recipe_sha256": recipe.sha256,
        "recipes": recipes, "recipe_manifest_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "chunks": list(chunks), "validation_boundaries": boundaries, "projection": projection,
        "panel_wall_budget_s": max(300., 3.*panel["wall_time_s"]),
        "input_files": dict(sorted(files.items())), "automatic_followup": False}
    return {**payload, "sha256": digest(payload)}


def engineering_stop(panels: list[dict]) -> dict:
    """Engineering diagnosis only; no promotion or production stage acceptance."""
    aligned = [p for p in panels if p["completed_iterations"] % INTERVAL == 0]
    recent = aligned[-4:]
    reason = None
    if len(recent) == 4 and all(b["completed_iterations"]-a["completed_iterations"] == INTERVAL
                                for a, b in zip(recent, recent[1:])):
        if all(p["launched_attempts"] == 0 for p in recent):
            reason = "four_panel_all_unlaunched_diagnosis"
        elif all(p.get("stopping_metrics") is not None for p in recent):
            if plateau([p["stopping_metrics"] for p in recent]):
                reason = "four_panel_physical_metric_plateau"
    return {"stop_learning": reason is not None, "stop_reason": reason,
            "production_accepted": False, "stage_accepted": False}


def _atomic_status(path: Path, payload):
    path = writable_path(path)
    temporary = path.with_name(path.name+".incomplete")
    write_json(temporary, payload)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.link(temporary, path)  # publish without overwriting a previous boundary
    temporary.unlink()


def coordinated_panel(runner, directory: Path, *, completed: int, expectation_identity: dict,
                      panel: callable, timeout_s: float) -> dict:
    """No outstanding NCCL call while rank0 constructs/runs a separate panel."""
    import torch
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("A finite explicit panel wall budget is required")
    rank, distributed = runner.gpu_global_rank, runner.is_distributed
    token = uuid.uuid4().hex if rank == 0 else None
    if distributed:
        token_list = [token]
        torch.distributed.broadcast_object_list(token_list, src=0)
        token = token_list[0]
        torch.distributed.barrier()
    identity = {"completed_iterations": completed, "nonce": token, **expectation_identity}
    destination = writable_path(directory/f"panel_{completed:06d}_{token}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if rank == 0:
        try:
            result = {"ok": True, "panel": panel()}
        except Exception as error:
            result = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        elapsed = time.monotonic()-started
        if elapsed > timeout_s:
            result = {"ok": False, "error": "Independent panel exceeded its explicit wall budget"}
        _atomic_status(destination, {"identity": identity, "elapsed_s": elapsed, **result})
    else:
        while not destination.exists():
            if time.monotonic()-started > timeout_s+30.:
                raise TimeoutError("Panel status was not published within the explicit bound")
            time.sleep(.25)
    result = json.loads(destination.read_text())
    if result.get("identity") != identity:
        raise ValueError("Panel status belongs to another boundary/actor/contract")
    if distributed:
        receipts = [None]*runner.gpu_world_size
        torch.distributed.all_gather_object(receipts, digest(result))
        if len(set(receipts)) != 1:
            raise RuntimeError("Ranks received different validation status")
    return result


def run_engineering_chunks(runner, *, iterations: int, completed: int, inherited_learn,
                           validate_boundary, checkpoint_boundary, histories=(), pending_boundary=None):
    """Only outer orchestration. Rollouts, PPO and normalization stay inherited."""
    panels = list(histories)
    for count in iteration_chunks(completed, iterations):
        env = runner.env.unwrapped
        before = env.common_step_counter
        runner.current_learning_iteration = completed
        inherited_learn(num_learning_iterations=count, init_at_random_ep_len=False)
        if (runner.current_learning_iteration != completed+count-1
                or env.common_step_counter-before != count*24):
            raise RuntimeError("Inherited completed iterations or fresh rollout count differs")
        completed += count
        if pending_boundary is not None:
            pending_boundary(completed, panels)
        try:
            result = validate_boundary(completed)
        except Exception as error:
            result = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        if result.get("ok"):
            panels.append(result["panel"])
            decision = engineering_stop(panels)
        else:
            decision = {"stop_learning": True, "stop_reason": "validation_failed",
                        "production_accepted": False, "stage_accepted": False}
        # Full learner state remains available even if the independent panel
        # failed. Validation failure never silently resumes further iterations.
        checkpoint_boundary(completed, panels, result, decision)
        if not result.get("ok"):
            raise RuntimeError("Engineering validation failed: "+result.get("error", "missing evidence"))
        if decision["stop_learning"]:
            return {"completed_iterations": completed, "panels": panels, **decision}
    return {"completed_iterations": completed, "panels": panels, "stop_learning": False,
            "stop_reason": "requested_budget_completed", "production_accepted": False, "stage_accepted": False}


def stopping_panel(summary: dict, completed: int) -> dict:
    """Retain actual event metrics; explicitly label penalties used for stopping.

    Unlaunched requests have no physical TTC error. The general plateau test is
    therefore used only for panels in which all requests launched; all-unlaunched
    panels have their own diagnosis rule. Miss/recovery penalties never replace
    measured/null physical values in the panel report.
    """
    episodes = summary["episodes"]
    if (summary.get("engineering_checkpoint_selection_eligible") is not True
            or not summary.get("all_terminal") or len(episodes) != 64
            or summary.get("attempts") != 64
            or summary.get("launched") != sum(row["launched"] for row in episodes)):
        raise ValueError("Stopping requires 64 independently verified terminal attempts")
    metrics = None
    if summary["launched"] == len(episodes):
        if any(row["outward_minimum_target_distance_m"] is None for row in episodes):
            raise ValueError("Launched panel lacks target-distance evidence")
        metrics = {
            "worst_family_success": min(row["strict_cycle_success"]/row["attempts"]
                for row in summary["families"].values()),
            "targeting_error": sum(max(0., row["outward_minimum_target_distance_m"]-.14)
                                   for row in episodes)/len(episodes),
            "timing_error": sum(abs(row["contact_ttc_error_s"]) if row["contact_ttc_error_s"] is not None
                                else 4. for row in episodes)/len(episodes),
            "recovery_time": sum(row["recovery_cycle_s"] if row["recovery_cycle_s"] is not None
                                 else row["requested_ttc_s"]+4. for row in episodes)/len(episodes)}
    return {"completed_iterations": completed, "launched_attempts": summary["launched"],
        "summary": summary, "stopping_metrics": metrics,
        "stopping_metric_policy": "Only all-launched panels: missing contact penalty4s; missing recovery penalty requestedTTC+4s. Physical report values remain null.",
        "production_accepted": False}


def _validate_native_boundary(runner, bundle, admission, plan, contract, recipes_path, output, completed):
    import torch
    from .engineering_cycle_panel import (EngineeringPanelExpectation, run_engineering_cycle_panel,
                                          verify_engineering_panel)
    from .trainedactor_evaluation import actor_identity, evaluation_contract, native_interfaces
    from .training_witness import training_state_witness, assert_training_state_unchanged
    from .validation import isolated_validation_actor
    destination = writable_path(output/f"panel_{completed:06d}")
    destination.mkdir(parents=True, exist_ok=False)
    before = training_state_witness(runner.env.unwrapped, runner=runner)
    write_json(destination/"training_before.json", before)
    report_path = None
    try:
        with isolated_validation_actor(runner) as (_, live_identity):
            # Inherited save exports every completed chunk. Load its actual JIT,
            # rather than assuming an in-memory conversion proves export parity.
            source = Path(runner.logger.log_dir)/"params/policy_jit.pt"
            actor = torch.jit.load(str(source), map_location=runner.device).eval()
            for parameter in actor.parameters():
                parameter.requires_grad_(False)
            identity = actor_identity(actor)
            if identity != live_identity:
                raise RuntimeError("Completed checkpoint export differs from the live actor/normalizer")
            # Preserve this boundary's export before the next inherited save.
            jit = destination/"policy_jit.pt"
            jit.write_bytes(source.read_bytes())
            actor_contract = evaluation_contract(actor, bundle, **native_interfaces(runner.env.unwrapped, bundle),
                checkpoint_contract_sha256=contract.sha256)
            expectation = EngineeringPanelExpectation(completed_iterations=completed, training_seed=plan["seed"],
                **identity, checkpoint_contract_sha256=contract.sha256,
                reference_manifest_sha256=plan["reference_manifest_sha256"],
                recipe_manifest_sha256=sha256_file(recipes_path), engineering_admission_sha256=admission.sha256,
                measurement_contract_sha256=plan["measurement_contract_sha256"], episodes_per_family=16)
            write_json(destination/"expectation.json", asdict(expectation))
            report_path = run_engineering_cycle_panel(bundle, actor, actor_contract, recipes_path,
                destination/"native", expectation=expectation, engineering_admission=admission)
            summary = verify_engineering_panel(report_path, recipes_path, expectation)
    finally:
        after = training_state_witness(runner.env.unwrapped, runner=runner)
        write_json(destination/"training_after.json", after)
        assert_training_state_unchanged(before, after)
    result = {**stopping_panel(summary, completed), "report": _evidence(report_path),
        "actor_export": _evidence(jit), "expectation": asdict(expectation),
        "training_before": _evidence(destination/"training_before.json"),
        "training_after": _evidence(destination/"training_after.json"), "training_state_unchanged": True}
    result["record"] = _evidence(write_json(destination/"verified.json", result))
    return result


def _engineering_worker(bundle_path: str, admission_values: dict, plan: dict, output: str) -> None:
    """Construct native WBC task once; orchestrate inherited PPO in fixed chunks."""
    import torch
    import warp as wp
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.scripts.train import TrainConfig
    from mjlab.utils.torch import configure_torch_backends
    from wbc_mjlab.rl.runner import PolicyOnlyMotionTrackingRunner
    from .capacity import _local_caches
    from .capacity_evidence import (engineering_train_config, attach_engineering_cycle,
                                   cycle_capacity_health, distributed_normalization_evidence,
                                   fresh_imitation_health_matches, initialization_health_matches)
    from .checkpointing import (BoundaryProgress, build_checkpoint_contract,
                               load_boundary_checkpoint, save_boundary_checkpoint)
    from .engineering_admission import EngineeringAdmission, verify_engineering_admission
    from .engineering_cycle_panel import write_engineering_recipes
    from .measurement_contract import gpu_hardware_identity, verify_gpu_hardware
    from .runtime.bundle import load_runtime_bundle
    from .runtime.runner import HumanMotionTrackingRunner
    from .runtime.task import register_human_task
    from .trainedactor_evaluation import native_interfaces
    recipe = _plan_recipe(plan)
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 2 or rank not in (0, 1):
        raise RuntimeError("Canonical engineering requires exactly two local learners")
    root = writable_path(output)
    destination = writable_path(root/f"rank_{rank:02d}")
    destination.mkdir(exist_ok=False)
    _local_caches(destination, shared_tmp=root/"cache/tmp")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    previous_directory = Path.cwd()
    os.chdir(destination)
    env, runner, sink, optimizer_hook = None, None, None, None
    record = {"schema": "human_wbc_engineering_training_rank_v1", "rank": rank,
        "status": "failed", "scope": SCOPE, "production_accepted": False,
        "planned_iterations": plan["iterations"], "seed": plan["seed"], "exception": None,
        "engineering_recipe": recipe.manifest(), "engineering_recipe_sha256": recipe.sha256}
    try:
        bundle = load_runtime_bundle(bundle_path)
        admission = EngineeringAdmission.from_dict(admission_values)
        verify_engineering_admission(bundle, admission)
        if not runtime_record()["declared_pins_match"]:
            raise RuntimeError("Native training runtime changed")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        configure_torch_backends()
        verify_gpu_hardware(plan["gpu_hardware"], gpu_hardware_identity(device))

        class EngineeringTrainingRunner(HumanMotionTrackingRunner):
            def load(self, *args, **kwargs):
                raise RuntimeError("Only the verified full engineering boundary loader may restore state")

            def learn(self, *args, **kwargs):
                verify_engineering_admission(bundle, admission)
                return PolicyOnlyMotionTrackingRunner.learn(self, *args, **kwargs)

            def save(self, path, infos=None):
                return super().save(path, {**(infos or {}), "engineering_training": {
                    "scope": SCOPE, "seed": plan["seed"], "scratch_lineage": True,
                    "admission_sha256": admission.sha256, "production_checkpoint": False,
                    "measurement_contract_sha256": plan["measurement_contract_sha256"],
                    "engineering_recipe": recipe.manifest(), "engineering_recipe_sha256": recipe.sha256}})

            def build_policy_export_metadata(self, run_path):
                return {**super().build_policy_export_metadata(run_path),
                    "learning_scope": SCOPE, "production_checkpoint": False,
                    "engineering_admission_sha256": admission.sha256,
                    "engineering_recipe": recipe.manifest(), "engineering_recipe_sha256": recipe.sha256}

        task_id = register_human_task(bundle, 4096, evaluation=True, runner_cls=EngineeringTrainingRunner)
        cfg = engineering_train_config(TrainConfig.from_task(task_id), bundle, admission,
            num_envs=4096, conditioning=plan["conditioning"], engineering_recipe=recipe.recipe_id)
        cfg.agent.seed = cfg.env.seed = plan["seed"]+rank
        cfg.agent.resume = False
        cfg.agent.max_iterations = plan["iterations"]
        cfg.agent.save_interval = MAX_ITERATIONS+1
        cfg.env.sim.nan_guard.enabled = True
        cfg.env.sim.nan_guard.output_dir = str(destination/"nan_guard")
        current = build_measurement_contract(bundle, scale=TrainingScale(), conditioning=plan["conditioning"],
            env_cfg=cfg.env, agent_cfg=cfg.agent, engineering_admission=admission,
            observer_report=read_artifact(plan["inputs"]["observer_report"]),
            bridge_report=read_artifact(plan["inputs"]["bridge_report"]),
            engineering_recipe=recipe.recipe_id,
            training_initialization_report=read_artifact(plan["inputs"]["training_initialization_report"])
                if "training_initialization_report" in plan["inputs"] else None)
        verify_measurement_contract(current, plan["measurement_contract"])
        stream = torch.cuda.ExternalStream(wp.get_stream(str(device)).cuda_stream, device=device)
        caller = torch.cuda.current_stream(device)
        stream.wait_stream(caller)
        with torch.cuda.stream(stream):
            env = ManagerBasedRlEnv(cfg=cfg.env, device=str(device))
            _, sink = attach_engineering_cycle(env, admission, output=destination/"attempts", measurement_contract=current)
            wrapped = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
            runner = EngineeringTrainingRunner(wrapped, asdict(cfg.agent), str(destination/"learner"), str(device))
            if (runner.gpu_world_size != 2 or env.num_envs != 4096 or runner.alg._raw_actor.obs_dim != 398):
                raise RuntimeError("Actual global batch or actor interface differs")
            interfaces = native_interfaces(env, bundle)
            sources = current["payload"]["sources"]["human_files"]
            learner = {name: asdict(cfg.agent)[name] for name in
                ("actor", "critic", "algorithm", "obs_groups", "num_steps_per_env", "clip_actions")}
            learner["engineering_recipe"] = recipe.manifest()
            learner["engineering_recipe_sha256"] = recipe.sha256
            contract = build_checkpoint_contract(body_path=bundle.model_path, controller=bundle.controller_contract(),
                actor_observation_schema=interfaces["observation_contract"], action_schema=interfaces["action_contract"],
                simulation_contract=interfaces["simulation_contract"], learner_configuration=learner,
                reference_manifest=bundle.root/"manifest.json",
                implementation_sources={name: PROJECT_ROOT/name for name in sources})
            write_json(destination/"session_contract.json", {"contract": asdict(contract), "sha256": contract.sha256,
                "measurement_contract_sha256": current["sha256"], "sampler_identity": sampler_identity(plan["seed"], recipe)})
            history, completed = [], plan["prior_completed_iterations"]
            if plan["resume"] is not None:
                resumed = plan["resume"]
                restoration = load_boundary_checkpoint(PROJECT_ROOT/resumed["path"], runner,
                    manifest_sha256=resumed["manifest_sha256"], expected_contract=contract, stage="canonical",
                    sampler_identity=sampler_identity(plan["seed"], recipe), reset_environment=wrapped.reset)
                history = restoration["progress"]["validation"]["panels"]
                write_json(destination/"restoration.json", restoration)
            recipes_path = root/"recipes.json"
            if rank == 0:
                write_engineering_recipes(recipes_path, plan["recipes"])
            torch.distributed.barrier(device_ids=[local_rank])
            if sha256_file(recipes_path) != plan["recipe_manifest_sha256"]:
                raise ValueError("Native validation recipe hash differs from the prepared plan")
            starting_policy_counter = env.common_step_counter
            count = {"steps": 0}
            def optimizer_step(*_):
                count["steps"] += 1
            optimizer_hook = runner.alg.optimizer.register_step_post_hook(optimizer_step)
            def inherited(**kwargs):
                before = count["steps"]
                value = runner.learn(**kwargs)
                if count["steps"]-before != kwargs["num_learning_iterations"]*20:
                    raise RuntimeError("Actual PPO optimizer count differs from five epochs/four minibatches")
                writer = runner.logger.writer
                if writer is not None:
                    writer.flush(); writer.close()
                    runner.logger.writer = None
                return value
            def checkpoint(done, panels, status, decision, *, pending=False):
                sink.flush()
                health = cycle_capacity_health(env, sink)
                if not health["terminal_accounting_consistent"]:
                    raise RuntimeError("Training terminal attempt accounting differs")
                if not initialization_health_matches(health, engineering_recipe=recipe.recipe_id):
                    raise RuntimeError("Training initialization provenance/counts differ from the measured recipe")
                if not fresh_imitation_health_matches(health.get("fresh_imitation", {}),
                        env.common_step_counter-starting_policy_counter):
                    raise RuntimeError("Raw WBC imitation rewards were nonfinite or omitted a physical window")
                normalization = distributed_normalization_evidence(runner, include_parameters=True)
                if not normalization["all_ranks_match"] or not normalization["normalization_finite"]:
                    raise RuntimeError("Distributed learned normalization differs or is nonfinite")
                progress = BoundaryProgress(done, "canonical", done, len(panels),
                    curriculum=training_curriculum(recipe),
                    validation={"scope": SCOPE, "panels": panels, "latest_status": status,
                        "decision": decision, "resume_allowed": not pending and status.get("ok") is True
                            and not decision.get("stop_learning", False)})
                name = f"boundary_{done:06d}"+("_pending_validation" if pending else "")
                receipt = save_boundary_checkpoint(root/name, runner, contract=contract,
                    progress=progress, sampler_identity=sampler_identity(plan["seed"], recipe))
                write_json(destination/(name+".json"), {"checkpoint": receipt, "cycle": health,
                    "normalization": normalization, "optimizer_steps_this_session": count["steps"]})
            def validate(done):
                return coordinated_panel(runner, root/"panel_status", completed=done,
                    expectation_identity={"contract_sha256": contract.sha256, "recipe_sha256": plan["recipe_manifest_sha256"]},
                    panel=lambda: _validate_native_boundary(runner, bundle, admission, plan, contract,
                        recipes_path, root/"validation", done), timeout_s=plan["panel_wall_budget_s"])
            record["result"] = run_engineering_chunks(runner, iterations=plan["iterations"], completed=completed,
                inherited_learn=inherited, validate_boundary=validate, checkpoint_boundary=checkpoint, histories=history,
                pending_boundary=lambda done, panels: checkpoint(done, panels, {"ok": False, "pending": True}, {}, pending=True))
            record.update(status="completed", actual_optimizer_steps=count["steps"],
                fresh_transitions_this_session=(record["result"]["completed_iterations"]-completed)*196608,
                completed_iterations_this_session=record["result"]["completed_iterations"]-completed)
        caller.wait_stream(stream)
    except Exception as error:
        record["exception"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if optimizer_hook is not None:
            optimizer_hook.remove()
        try:
            if sink is not None:
                sink.flush()
                record["terminal_attempts"] = sink.report()
            if env is not None:
                env.close()
        except Exception as error:
            record.update(status="failed", close_exception={"type": type(error).__name__, "message": str(error)})
            raise
        finally:
            write_json(destination/"session.json", record)
            os.chdir(previous_directory)


def run_engineering_training(bundle, admission_path, observer_report, bridge_report, capacity_report,
                             output, *, iterations, seed, resume=None, resume_manifest_sha256=None,
                             compute_plan=None, engineering_recipe=LEGACY_RECIPE_ID,
                             training_initialization_report=None):
    plan = checked_engineering_plan(bundle, admission_path, observer_report, bridge_report, capacity_report,
        iterations=iterations, seed=seed, resume=resume, resume_manifest_sha256=resume_manifest_sha256,
        engineering_recipe=engineering_recipe, training_initialization_report=training_initialization_report)
    if compute_plan is not None and json.loads(Path(compute_plan).read_text()) != plan:
        raise ValueError("Executing engineering plan differs from the prepared packet")
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Engineering learning requires an existing allocated Slurm job")
    import torch
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.cuda.device_count() != 2:
        raise RuntimeError("Launch once inside an allocation with exactly two visible L40S GPUs")
    from .capacity import _local_caches
    from .capacity_evidence import load_engineering_admission
    import torchrunx
    destination = writable_path(output)
    destination.mkdir(parents=True, exist_ok=False)
    _local_caches(destination, shared_tmp=destination/"cache/tmp")
    write_json(destination/"launch.json", {"schema": "human_wbc_engineering_training_launch_v1",
        "plan": plan, "slurm_job_id": os.environ["SLURM_JOB_ID"], "runtime": runtime_record(),
        "automatic_followup": False, "production_accepted": False})
    os.environ["TORCHRUNX_LOG_DIR"] = str(destination/"torchrunx")
    torchrunx.Launcher(hostnames=["localhost"], workers_per_host=2, backend=None,
        copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY+(
            "MUJOCO*", "WARP*", "SLURM_JOB_ID", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "TMPDIR",
            "MPLCONFIGDIR", "TORCH_HOME", "CUDA_CACHE_PATH", "PYTHONDONTWRITEBYTECODE")).run(
                _engineering_worker, str(Path(bundle).resolve(strict=True)),
                load_engineering_admission(admission_path).as_dict(), plan, str(destination))
    ranks = [json.loads((destination/f"rank_{rank:02d}"/"session.json").read_text()) for rank in range(2)]
    if any(row["status"] != "completed" for row in ranks) or ranks[0]["result"] != ranks[1]["result"]:
        raise RuntimeError("Engineering session ranks did not finish the same verified boundary")
    return write_json(destination/"session_complete.json", {
        "schema": "human_wbc_engineering_training_complete_v1", "scope": SCOPE,
        "result": ranks[0]["result"], "plan_sha256": plan["sha256"],
        "ranks": [_evidence(destination/f"rank_{rank:02d}"/"session.json") for rank in range(2)],
        "production_accepted": False, "automatic_followup": False})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "engineering-admission", "observer-report", "bridge-report", "capacity-report", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-manifest-sha256")
    parser.add_argument("--compute-plan", type=Path)
    parser.add_argument("--engineering-recipe", choices=RECIPE_IDS, default=LEGACY_RECIPE_ID)
    parser.add_argument("--training-initialization-report", type=Path)
    args = parser.parse_args(argv)
    result = run_engineering_training(args.bundle, args.engineering_admission, args.observer_report,
        args.bridge_report, args.capacity_report, args.output, iterations=args.iterations, seed=args.seed,
        resume=args.resume, resume_manifest_sha256=args.resume_manifest_sha256, compute_plan=args.compute_plan,
        engineering_recipe=args.engineering_recipe, training_initialization_report=args.training_initialization_report)
    print(json.dumps({"completed_session": str(result), "production_accepted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
