"""Separate terminal frozen-actor diagnostics for disposable capacity learners."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

from .engineering_admission import _evidence, read_artifact
from .provenance import sha256_file, writable_path, write_json


def coexistence_memory(before: dict, after: dict, native: dict) -> dict:
    """Combine panel-native samples with resident-learner Torch peaks.

    The non-Torch sample includes resident Warp/simulator allocations. Its
    maximum need not coincide with the Torch maximum, so their sum is a
    conservative estimate, not an exact continuous allocation measurement.
    """
    try:
        total = before["total_bytes"]
        amounts = (native["conservative_peak_used_bytes"],
            after["torch_peak_reserved_bytes"], native["maximum_sampled_non_torch_bytes"],
            before["driver_used_bytes"], after["driver_used_bytes"])
        if (type(total) is not int or total <= 0 or native.get("available") is not True
                or native.get("total_vram_bytes") != total
                or any(type(value) is not int or value < 0 for value in amounts)):
            raise ValueError("Invalid native coexistence memory samples")
        peak = max(amounts[0], amounts[1]+amounts[2], amounts[3], amounts[4])
        if peak <= 0:
            raise ValueError("Missing resident learner memory")
    except (KeyError, TypeError) as error:
        raise ValueError("Missing native coexistence memory evidence") from error
    return {"coexistence_peak_used_bytes": peak, "total_vram_bytes": total,
            "headroom_fraction": 1-peak/total}


def prepare_terminal_panel(runner, destination: Path, capacity_record: dict) -> None:
    """Persist completed measurements before a short collective NCCL teardown."""
    import torch
    if capacity_record.get("status") != "passed":
        raise ValueError("Only completed capacity phases may enter a post-measurement panel")
    write_json(destination/"capacity_completed.json", capacity_record)
    if runner.is_distributed:
        torch.distributed.barrier(device_ids=[torch.device(runner.device).index])
        torch.distributed.destroy_process_group()


def run_post_capacity_panel(runner, bundle, admission, measurement_contract, config,
                            capacity_record, recipes_path: Path, output: Path) -> dict:
    """No further PPO, no production selection, and no distributed operations.

    The caller must persist all completed capacity records and collectively
    terminate NCCL before entering this rank-zero-only function. Native training
    allocations stay resident so memory and caller-state parity are measured.
    Errors are retained separately from the already completed capacity phase.
    """
    import torch
    from .engineering_cycle_panel import (EngineeringPanelExpectation, run_engineering_cycle_panel,
                                          verify_engineering_panel)
    from .trainedactor_evaluation import actor_identity, evaluation_contract, native_interfaces
    from .training_witness import training_state_witness, assert_training_state_unchanged
    from .validation import isolated_validation_actor
    from .runtime.bundle import load_runtime_bundle
    bundle = load_runtime_bundle(bundle)
    if torch.distributed.is_initialized():
        raise RuntimeError("Terminal capacity panel requires completed collective process-group teardown")
    destination = writable_path(output)
    destination.mkdir(parents=True, exist_ok=False)
    device, env = runner.device, runner.env.unwrapped
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    report = {"schema": "human_wbc_post_capacity_panel_v1", "status": "failed",
        "capacity_phase_already_completed": True, "capacity_measurement_unchanged": True,
        "capacity_weights_may_initialize_training": False, "production_acceptance": False,
        "training_allocations_remain_resident": True, "ppo_updates": 0,
        "distributed_group_destroyed": True, "training_state_unchanged": False,
        "memory_before": {"total_bytes": int(total), "driver_used_bytes": int(total-free),
            "torch_reserved_bytes": int(torch.cuda.memory_reserved(device))},
        "exception": None}
    before = after = None
    try:
        before = training_state_witness(env, runner=runner)
        before_path = write_json(destination/"training_before.json", before)
        report["training_before"] = _evidence(before_path)
        report["recipes"] = _evidence(recipes_path)
        iterations = config.warmup_iterations+config.measurement_iterations
        checkpoint = Path(runner.logger.log_dir)/f"model_{iterations-1}.pt"
        jit = Path(runner.logger.log_dir)/"params/policy_jit.pt"
        checkpoint_contract = {
            "schema": "disposable_capacity_checkpoint_contract_v1", "capacity_only": True,
            "capacity_weights_may_initialize_training": False, "checkpoint": _evidence(checkpoint),
            "jit_export": _evidence(jit), "measurement_contract_sha256": measurement_contract["sha256"],
            "completed_iterations": iterations, "global_scale": config.scale.report(iterations),
            "observed_rank": capacity_record["rank"],
            "observed_warmup_iterations": capacity_record["warmup_iterations"],
            "observed_measurement_iterations": capacity_record["measurement_iterations"],
            "reference_and_body": measurement_contract["payload"]["reference_and_body"],
            "sources": measurement_contract["payload"]["sources"]}
        contract_path = write_json(destination/"checkpoint_contract.json", checkpoint_contract)
        with isolated_validation_actor(runner) as (_, live_identity):
            actor = torch.jit.load(str(jit), map_location=device).eval()
            for parameter in actor.parameters():
                parameter.requires_grad_(False)
            identity = actor_identity(actor)
            if identity != live_identity:
                raise RuntimeError("Saved capacity JIT differs from the live frozen actor/normalizer")
            actor_contract = evaluation_contract(actor, bundle, **native_interfaces(env, bundle),
                checkpoint_contract_sha256=sha256_file(contract_path))
            expectation = EngineeringPanelExpectation(completed_iterations=iterations,
                training_seed=config.seed, **identity,
                checkpoint_contract_sha256=sha256_file(contract_path),
                reference_manifest_sha256=capacity_record["reference_manifest_sha256"],
                recipe_manifest_sha256=sha256_file(recipes_path),
                engineering_admission_sha256=admission.sha256,
                measurement_contract_sha256=measurement_contract["sha256"], episodes_per_family=16)
            write_json(destination/"expectation.json", asdict(expectation))
            panel_path = run_engineering_cycle_panel(bundle, actor, actor_contract, recipes_path,
                destination/"panel", expectation=expectation, engineering_admission=admission)
            summary = verify_engineering_panel(panel_path, recipes_path, expectation)
            report.update(panel_report=_evidence(panel_path), verified_summary=summary,
                expectation=asdict(expectation), checkpoint_contract=_evidence(contract_path),
                actor_identity=identity, panel_native_memory=json.loads(panel_path.read_text()).get("memory"))
        if sha256_file(checkpoint) != checkpoint_contract["checkpoint"]["sha256"] or sha256_file(jit) != checkpoint_contract["jit_export"]["sha256"]:
            raise RuntimeError("Post-capacity panel changed the checkpoint or exported actor")
        report["status"] = "passed"
    except Exception as error:
        report["exception"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if before is not None:
            try:
                after = training_state_witness(env, runner=runner)
                after_path = write_json(destination/"training_after.json", after)
                report["training_after"] = _evidence(after_path)
                assert_training_state_unchanged(before, after)
                report["training_state_unchanged"] = True
            except Exception as error:
                report.update(status="failed", state_witness_exception={
                    "type": type(error).__name__, "message": str(error)})
        torch.cuda.synchronize(device)
        final_free, _ = torch.cuda.mem_get_info(device)
        report.update(wall_time_s=time.perf_counter()-started,
            memory_after={"driver_used_bytes": int(total-final_free),
                "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))},
            memory_scope="Native panel samples plus Torch peaks, including resident capacity allocations; driver samples are not a continuous external-allocation peak")
        try:
            report.update(coexistence_memory(report["memory_before"], report["memory_after"],
                                             report.get("panel_native_memory") or {}))
        except ValueError as error:
            report.update(coexistence_peak_used_bytes=None, total_vram_bytes=int(total), headroom_fraction=None)
            report.update(status="failed", memory_evidence_exception={
                "type": type(error).__name__, "message": str(error)})
        receipt_path = write_json(destination/"receipt.json", report)
    return {**report, "report": _evidence(receipt_path)}


def verify_post_measurement_panel(entry: dict) -> dict:
    """Re-read native panel, saved actor and caller-state evidence for planning."""
    import torch
    from .engineering_cycle_panel import EngineeringPanelExpectation, verify_engineering_panel
    from .trainedactor_evaluation import actor_identity
    from .training_witness import assert_training_state_unchanged
    from .measurement_contract import digest
    receipt = json.loads(read_artifact(entry["report"]).read_text())
    if entry != {**receipt, "report": entry["report"]}:
        raise ValueError("Aggregate panel entry differs from its immutable receipt")
    if (receipt.get("schema") != "human_wbc_post_capacity_panel_v1" or receipt.get("status") != "passed"
            or receipt.get("capacity_phase_already_completed") is not True
            or receipt.get("training_allocations_remain_resident") is not True
            or receipt.get("training_state_unchanged") is not True
            or receipt.get("capacity_weights_may_initialize_training") is not False
            or receipt.get("ppo_updates") != 0):
        raise ValueError("Completed independent post-capacity panel evidence is required")
    before = json.loads(read_artifact(receipt["training_before"]).read_text())
    after = json.loads(read_artifact(receipt["training_after"]).read_text())
    assert_training_state_unchanged(before, after)
    if "learner" not in before["fields"]:
        raise ValueError("Panel coexistence witness omits the live actor/optimizer state")
    contract_path = read_artifact(receipt["checkpoint_contract"])
    contract = json.loads(contract_path.read_text())
    expected = EngineeringPanelExpectation(**receipt["expectation"])
    if (contract.get("schema") != "disposable_capacity_checkpoint_contract_v1"
            or contract.get("capacity_only") is not True
            or contract.get("capacity_weights_may_initialize_training") is not False
            or contract.get("completed_iterations") != expected.completed_iterations
            or contract.get("observed_warmup_iterations", 0)+contract.get("observed_measurement_iterations", 0)
                != expected.completed_iterations
            or contract.get("measurement_contract_sha256") != expected.measurement_contract_sha256
            or sha256_file(contract_path) != expected.checkpoint_contract_sha256
            or contract["reference_and_body"].get("reference_manifest_sha256") != expected.reference_manifest_sha256
            or contract["reference_and_body"].get("engineering_admission_sha256") != expected.engineering_admission_sha256):
        raise ValueError("Disposable checkpoint contract differs from the evaluated actor")
    checkpoint_path, jit_path = read_artifact(contract["checkpoint"]), read_artifact(contract["jit_export"])
    actor = torch.jit.load(str(jit_path), map_location="cpu").eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    identity = actor_identity(actor)
    if (identity != receipt["actor_identity"] or identity["policy_sha256"] != expected.policy_sha256
            or identity["normalization_sha256"] != expected.normalization_sha256):
        raise ValueError("Frozen saved policy or normalization identity differs")
    # These are locally produced, hash-bound RSL checkpoints; no external pickle
    # source is accepted. Verify inference tensors against the saved learner.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = checkpoint.get("actor_state_dict", {})
    scope = (checkpoint.get("infos") or {}).get("engineering_capacity", {})
    if (checkpoint.get("iter", -2)+1 != expected.completed_iterations
            or scope.get("capacity_weights_may_initialize_training") is not False
            or scope.get("admission_sha256") != expected.engineering_admission_sha256
            or any(name not in raw or not torch.equal(value.cpu(), raw[name].cpu())
                for name, value in actor.state_dict().items())):
        raise ValueError("Saved PPO checkpoint and exported frozen actor disagree")
    panel_path, recipes_path = read_artifact(receipt["panel_report"]), read_artifact(receipt["recipes"])
    if sha256_file(recipes_path) != expected.recipe_manifest_sha256:
        raise ValueError("Predeclared panel recipe hash differs")
    summary = verify_engineering_panel(panel_path, recipes_path, expected)
    if digest(summary) != digest(receipt["verified_summary"]):
        raise ValueError("Native panel re-audit differs from the retained summary")
    native = json.loads(panel_path.read_text())
    memory = native.get("memory", {})
    if memory != receipt.get("panel_native_memory"):
        raise ValueError("Native panel memory evidence differs")
    recomputed = coexistence_memory(receipt["memory_before"], receipt["memory_after"], memory)
    if any(receipt.get(key) != value for key, value in recomputed.items()):
        raise ValueError("Coexistence memory calculation differs from native samples")
    records = {}
    def collect(value):
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                read_artifact(value)
                if value["path"] in records and records[value["path"]] != value["sha256"]:
                    raise ValueError("Conflicting panel artifact hashes")
                records[value["path"]] = value["sha256"]
            for child in value.values(): collect(child)
        elif isinstance(value, list):
            for child in value: collect(child)
    collect(entry); collect(contract); collect(summary)
    return {"entry": entry, "input_files": [{"path": path, "sha256": records[path]} for path in sorted(records)]}
