"""Native observer and distributed-normalization evidence for engineering capacity."""
from __future__ import annotations

import json
from pathlib import Path

from .engineering_admission import EngineeringAdmission, _evidence, read_artifact
from .measurement_contract import digest
from .provenance import PROJECT_ROOT, sha256_file
from .runtime.bundle import load_runtime_bundle
from .runtime.engineering_recipe import (LEGACY_RECIPE_ID, resolve_engineering_recipe, verify_recipe_manifest)

OBSERVER_REQUIRED_SOURCES = (
    "src/human_punching/wbc/telemetry/kinematics.py",
    "src/human_punching/wbc/telemetry/contacts.py",
    "src/human_punching/wbc/telemetry/streams.py",
    "src/human_punching/wbc/telemetry/runtime_telemetry.py",
    "src/human_punching/wbc/telemetry/native_qualification.py",
    "src/human_punching/wbc/telemetry/posthoc_qualification.py",
    "src/human_punching/wbc/telemetry/cycle_qualification.py",
    "src/human_punching/wbc/runtime/action.py", "src/human_punching/wbc/runtime/body.py",
    "src/human_punching/wbc/runtime/command.py")


def load_engineering_admission(path: Path) -> EngineeringAdmission:
    return EngineeringAdmission.from_dict(json.loads(read_artifact(_evidence(path)).read_text()))


def observer_run_payloads(directory: Path, row: dict) -> dict:
    """Bind the recorded arrays and the compiled model used for CPU re-audit."""
    payloads = {}
    for name, expected in row["hashes"].items():
        record = _evidence(directory/(name+".npz"))
        if record["sha256"] != expected:
            raise ValueError("Observer native trace hash differs")
        payloads[name] = record
    compiled = _evidence(directory/"compiled_scene.mjb")
    if compiled["sha256"] != row.get("compiled_scene_sha256"):
        raise ValueError("Observer compiled scene hash differs")
    payloads["compiled_scene"] = compiled
    return payloads


def observer_capacity_evidence(bundle, report_path: Path) -> dict:
    """Require instrumentation health, retaining strict rollout comparison failures.

    Native GPU trajectory repeatability is not a prerequisite for learning. An
    incomplete observer run, live-state mutation, stale FK/contact decoding or
    clock/force failure still blocks this entry point. No failed qualification is
    rewritten as passed, and these diagnostics do not establish punch acceptance.
    """
    report_record = _evidence(report_path)
    path = read_artifact(report_record)
    report = json.loads(path.read_text())
    posthoc = report.get("schema") == "human_wbc_observer_posthoc_qualification_v1"
    execution_record = report["original_execution"] if posthoc else _evidence(path.parent.parent/"execution.json")
    packet_record = report["original_packet"] if posthoc else _evidence(path.parent.parent/"packet.json")
    execution = json.loads(read_artifact(execution_record).read_text())
    packet = json.loads(read_artifact(packet_record).read_text())
    if (report.get("schema") not in ("human_wbc_observer_qualification_v1", "human_wbc_observer_posthoc_qualification_v1")
            or report.get("status") not in ("passed", "instrumentation_passed_rollouts_nondeterministic")
            or report.get("instrumentation_status") != "passed" or report.get("training") is not False
            or report.get("runtime", {}).get("declared_pins_match") is not True
            or execution.get("schema") != "human_wbc_telemetry_execution_v1"
            or execution.get("returncode") not in (0, 2) or execution.get("budget_stop") is not None
            or not execution.get("slurm_job_id") or execution.get("packet_sha256") != packet_record["sha256"]
            or packet.get("schema") != "human_wbc_telemetry_packet_v1"):
        raise ValueError("Observer instrumentation evidence is incomplete or has invalid provenance")
    checked = load_runtime_bundle(bundle)
    observed = load_runtime_bundle(PROJECT_ROOT/packet["bundle"])
    manifest_hash = report["bundle_manifest"]["sha256"] if posthoc else report.get("bundle_manifest_sha256")
    if (sha256_file(observed.root/"manifest.json") != manifest_hash
            or sha256_file(checked.model_path) != sha256_file(observed.model_path)
            or checked.controller_sha256() != observed.controller_sha256()):
        raise ValueError("Observer evidence uses a different body or controller")
    source_hashes, extra_provenance = {}, {}
    if posthoc:
        for name in ("original_report", "verifier_source", "bundle_manifest"):
            read_artifact(report[name])
            extra_provenance[name] = report[name]
        for relative, expected in report["verified_physical_reference_files"].items():
            read_artifact({"path": relative, "sha256": expected})
    for relative, expected in packet["files"].items():
        if relative in OBSERVER_REQUIRED_SOURCES:
            read_artifact({"path": relative, "sha256": expected})
            source_hashes[relative] = expected
    if set(source_hashes) != set(OBSERVER_REQUIRED_SOURCES):
        raise ValueError("Observer source identity missing from native packet")
    fixtures, payloads = {}, {}
    if set(report.get("fixtures", {})) != {"guard", "punch"}:
        raise ValueError("Both native guard and punch observer fixtures must complete")
    for kind, fixture in report["fixtures"].items():
        runs = fixture.get("runs", [])
        if (len(runs) != 3 or any(row.get("policy_steps") != 300 or row.get("physics_samples") != 1201
                or row.get("controller_audit" if posthoc else "native_controller_audit", {}).get("status") != "passed"
                or row.get("exception") is not None for row in runs)
                or runs[2].get("nonmutation", {}).get("passed") is not True
                or runs[2].get("offline_parity", {}).get("passed") is not True
                or (runs[2].get("stream_capture_roundtrip") or {}).get("passed") is not True):
            raise ValueError("Native observer controller, nonmutation or FK/contact parity failed")
        for label, row in zip(("a", "a_repeat", "b"), runs, strict=True):
            if posthoc:
                for name, record in row["inputs"].items():
                    read_artifact(record)
                    payloads[f"{kind}_{label}/{name}"] = record
            else:
                for name, record in observer_run_payloads(path.parent/f"{kind}_{label}", row).items():
                    payloads[f"{kind}_{label}/{name}"] = record
        fixtures[kind] = {"instrumentation_health_passed": True,
            "baseline_determinism": fixture["baseline_determinism"],
            "observer_comparison": fixture["observer_comparison"],
            "strict_qualification_passed": bool(fixture["baseline_determinism"]["passed"]
                                                  and fixture["observer_comparison"]["passed"]),
            "offline_parity": runs[2]["offline_parity"]}
    return {"report": report_record, "execution": execution_record, "packet": packet_record,
        "source_hashes": source_hashes, "trace_payloads": payloads, "fixtures": fixtures,
        "verified_physical_reference_files": report.get("verified_physical_reference_files", {}),
        "native_report_status": report["status"], "strict_rollout_status": report["strict_rollout_status"],
        "instrumentation_health_passed": True,
        "strict_trajectory_repeatability_required_for_learning": False,
        "additional_provenance": extra_provenance, "limitations": report.get("limitations", []),
        "acceptance_qualified": False, "production_throughput_measured": False}


def bridge_capacity_evidence(bundle, report_path: Path) -> dict:
    """Bind completed native bridge diagnostics without inventing launch evidence."""
    record = _evidence(report_path)
    path = read_artifact(record)
    report = json.loads(path.read_text())
    execution_record = _evidence(path.parent.parent/"execution.json")
    packet_record = _evidence(path.parent.parent/"packet.json")
    execution = json.loads(read_artifact(execution_record).read_text())
    packet = json.loads(read_artifact(packet_record).read_text())
    _require_bridge_report(report, execution, packet, packet_record["sha256"])
    checked = load_runtime_bundle(bundle)
    if sha256_file(checked.root/"manifest.json") != report["bundle_manifest_sha256"]:
        raise ValueError("Native bridge evidence uses different references or controller")
    runtime_files = {"action.py", "body.py", "bundle.py", "command.py", "mdp.py", "task.py",
                     "task_rewards.py", "task_reward_adapter.py", "training_initialization.py", "engineering_recipe.py"}
    sources = {name: expected for name, expected in packet["files"].items()
        if name.startswith("src/human_punching/wbc/telemetry/")
        or (name.startswith("src/human_punching/wbc/runtime/") and Path(name).name in runtime_files)}
    for required in ("telemetry/cycle_runtime.py", "telemetry/snapshot_providers.py",
                     "telemetry/attempt_receipts.py", "runtime/task_reward_adapter.py",
                     "runtime/training_initialization.py", "runtime/engineering_recipe.py"):
        if "src/human_punching/wbc/"+required not in sources:
            raise ValueError("Native bridge source identity is incomplete")
    for relative, expected in sources.items():
        read_artifact({"path": relative, "sha256": expected})
    payloads = {}
    if set(report["hashes"]) != {"native", "pre_interval", "endpoint", "policy_boundary"}:
        raise ValueError("Native bridge trace payload set is incomplete")
    for name, expected in report["hashes"].items():
        item = _evidence(path.parent/(name+".npz"))
        if item["sha256"] != expected:
            raise ValueError("Native bridge trace payload hash differs")
        payloads[name] = item
    payloads["compiled_scene"] = _evidence(path.parent/"compiled_scene.mjb")
    return {"report": record, "execution": execution_record, "packet": packet_record,
        "source_hashes": sources, "trace_payloads": payloads,
        "instrumentation_health_passed": True, "readiness": report["readiness"],
        "launch_observed": report["audit"]["launch_observed"],
        "launch_qualification": report["audit"]["launch_qualification"],
        "native_aggregate_attempts": report["aggregate_attempts"],
        "acceptance_qualified": False,
        "limitations": ["Zero-residual diagnostic is not learned cycle acceptance.",
            "Unlaunched capacity measures readiness, reset, telemetry and PPO workload; launched-cycle throughput remains unmeasured.",
            "Isolated immutable G1 snapshot providers; no moving G1 entity."]}


def _require_bridge_report(report, execution, packet, packet_hash):
    audit = report.get("audit", {})
    checks = audit.get("checks", {})
    required = {"complete_native_samples", "complete_policy_samples", "ordinary_reset_exercised",
        "one_initialization_per_reset", "fresh_endpoint_fk", "fresh_actor_anchor", "finite_task_rewards",
        "finite_observations", "zero_reference_before_launch", "future_held_before_launch"}
    steps = packet.get("policy_steps")
    if (report.get("schema") != "human_wbc_cycle_native_qualification_v1"
            or report.get("status") != "passed" or audit.get("status") != "passed"
            or not required <= checks.keys() or not all(value is True for value in checks.values())
            or report.get("training") is not False or report.get("exception") is not None
            or report.get("runtime", {}).get("declared_pins_match") is not True
            or report.get("readiness") != {"quiet_ticks": 40, "maximum_settling_ticks": 1200}
            or report.get("support_artifact_sha256") is not None
            or report.get("actor_observation_dimension") != [398]
            or type(steps) is not int or not 300 <= steps <= 1000
            or report.get("requested_policy_steps") != steps or report.get("completed_policy_steps") != steps
            or report.get("actual_physics_samples") != steps*4 or report.get("environments") != 16
            or execution.get("schema") != "human_wbc_cycle_qualification_execution_v1"
            or execution.get("returncode") != 0 or execution.get("budget_stop") is not None
            or not execution.get("slurm_job_id") or execution.get("packet_sha256") != packet_hash
            or packet.get("schema") != "human_wbc_cycle_qualification_packet_v1"
            or type(audit.get("launch_observed")) is not bool
            or audit.get("launch_checks_exercised") != audit.get("launch_observed")
            or audit.get("launch_qualification") != ("diagnostic_only" if audit.get("launch_observed") else "not_tested_unready")):
        raise ValueError("Completed native bridge instrumentation evidence with default readiness is required")


def configure_engineering_capacity(bundle, admission, *, num_envs, conditioning="reference_commands_frozen_g1",
                                   engineering_recipe=LEGACY_RECIPE_ID):
    from .conditioning import Conditioning, apply_conditioning
    from .runtime.training_initialization import TrainingInitializationCfg
    from .runtime.task_rewards import DEFAULT_WEIGHTS
    recipe = resolve_engineering_recipe(engineering_recipe)
    from .engineering_admission import build_engineering_env_cfg
    from .telemetry.cycle_runtime import configure_cycle_command
    from .telemetry.runtime_telemetry import ObserverOptions, configure_observer
    if Conditioning(conditioning) is not Conditioning.COMBINED:
        raise ValueError("Engineering cycle capacity requires the combined phase-aware actor")
    checked = load_runtime_bundle(bundle)
    ttcs = [row["requested_ttc_s"] for row in checked.clips]
    import math
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value <= 0 for value in ttcs):
        raise ValueError("Cycle capacity requires explicit finite positive requested TTCs")
    cfg = build_engineering_env_cfg(checked, admission, num_envs=num_envs, auto_reset=True)
    apply_conditioning(cfg, conditioning)  # Before bridge; never overwrite its phase features.
    configure_observer(cfg, ObserverOptions(enabled=True))
    configure_cycle_command(cfg, maximum_ttc_s=max(ttcs),
        require_physical_qualification=False, task_objectives=True,
        training_initialization=TrainingInitializationCfg(**recipe.training.initialization_parameters()))
    for name, weight in recipe.training.task_reward_weights(DEFAULT_WEIGHTS).items():
        term = cfg.rewards["human_"+name]
        term.weight = weight
        if recipe.training.guard_joint_posture_metric != "rms":
            term.params = {"name": name, "guard_joint_posture_metric": recipe.training.guard_joint_posture_metric}
    from .runtime.imitation_rewards import configure_fresh_imitation_rewards
    configure_fresh_imitation_rewards(cfg)
    return cfg


def engineering_train_config(train_cfg, bundle, admission, *, num_envs,
                             conditioning="reference_commands_frozen_g1", engineering_recipe=LEGACY_RECIPE_ID):
    """Replace the frozen pinned launcher config, preserving its mutable agent."""
    from dataclasses import replace
    from mjlab.scripts.train import TrainConfig
    if type(train_cfg) is not TrainConfig:
        raise TypeError("Expected the pinned mjlab TrainConfig")
    return replace(train_cfg, env=configure_engineering_capacity(bundle, admission,
        num_envs=num_envs, conditioning=conditioning, engineering_recipe=engineering_recipe))


def require_engineering_configuration(cfg, *, engineering_recipe=LEGACY_RECIPE_ID) -> None:
    from .telemetry.cycle_runtime import CycleObserverActionCfg, CycleMotionCommandCfg, cycle_condition, cycle_terminal
    from .runtime.task_reward_adapter import native_task_reward
    from .runtime.task_rewards import DEFAULT_WEIGHTS
    from .runtime.imitation_rewards import require_fresh_imitation_rewards
    from .runtime.training_initialization import TrainingInitializationCfg
    recipe = resolve_engineering_recipe(engineering_recipe)
    settings = recipe.training
    expected_initialization = TrainingInitializationCfg(**settings.initialization_parameters())
    weights = settings.task_reward_weights(DEFAULT_WEIGHTS)
    def expected_params(name):
        return ({"name": name} if settings.guard_joint_posture_metric == "rms" else
            {"name": name, "guard_joint_posture_metric": settings.guard_joint_posture_metric})
    require_fresh_imitation_rewards(cfg)
    motion = cfg.commands["motion"]
    metric = cfg.metrics.get("human_native_substep_valid")
    if (cfg.auto_reset is not True or type(cfg.actions["joint_pos"]) is not CycleObserverActionCfg
            or type(motion) is not CycleMotionCommandCfg or motion.require_physical_qualification is not False
            or metric is None or not metric.params["options"].enabled or not metric.per_substep
            or motion.training_initialization != expected_initialization
            or motion.assistive_wrench_enabled or motion.rsi.sampling_mode != "start" or cfg.events
            or any(cfg.observations[group].terms.get("punch_condition") is None
                or cfg.observations[group].terms["punch_condition"].func is not cycle_condition
                for group in ("actor", "critic"))
            or cfg.terminations.get("cycle_complete") is None
            or cfg.terminations["cycle_complete"].func is not cycle_terminal
            or any("human_"+name not in cfg.rewards or cfg.rewards["human_"+name].func is not native_task_reward
                or cfg.rewards["human_"+name].weight != weight
                or cfg.rewards["human_"+name].params != expected_params(name)
                for name, weight in weights.items())):
        raise ValueError("Engineering capacity requires explicit unassisted recipe initialization, episode reset, native observer, cycle bridge and task objectives")


def cycle_capacity_health(env, sink) -> dict:
    """Fixed-size phase/family/reset counts; never serialize full simulator state."""
    import torch
    from dataclasses import asdict
    from .telemetry.cycle_runtime import get_cycle_runtime
    from .telemetry.runtime_telemetry import get_observer
    from .telemetry.readiness_clock import Phase
    from .runtime.imitation_rewards import fresh_imitation_health
    bridge, observer = get_cycle_runtime(env), get_observer(env)
    r, command = bridge.readiness, env.command_manager.get_term("motion")
    values = torch.stack((r.attempt_count.sum(), r.launch_count.sum(), r.failure_count.sum(),
        r.complete_count.sum(), observer.reset_count.sum(), observer.reset_count.min(),
        observer.reset_count.max(), bridge.pending_reset.sum(),
        ((r.phase == Phase.SETTLING) | (r.phase == Phase.STRIKE) | (r.phase == Phase.RECOVERY)).sum(),
        bridge.training_initializer.reference_starts_count.sum(),
        bridge.training_initializer.grace_total_ticks.sum()))
    values = values.cpu().tolist()
    phases = torch.bincount(r.phase, minlength=len(Phase)).cpu().tolist()
    families = torch.bincount(command.family_ids[command.trajectory_ids], minlength=4).cpu().tolist()
    attempts = sink.report()
    return {"schema": "human_wbc_capacity_cycle_health_v1", "acceptance_qualified": False,
        "cumulative_attempts": values[0], "cumulative_launches": values[1],
        "cumulative_failures": values[2], "cumulative_recoveries": values[3],
        "cumulative_strict_ready_launches": values[1]-values[9],
        "cumulative_training_reference_starts": values[9],
        "cumulative_strict_guard_starts": values[0]-values[9],
        "cumulative_training_support_grace_ticks": values[10],
        "launch_count_semantics": "all_reference_starts_including_training_initialization",
        "training_initialization": asdict(bridge.training_initializer.config),
        "ordinary_episode_initializations": values[4], "reset_count_min": values[5], "reset_count_max": values[6],
        "pending_physical_initializations": values[7], "active_inflight_attempts": values[8],
        "current_phase_counts": {entry.name.lower(): phases[int(entry)] for entry in Phase},
        "current_family_counts": families, "terminal_receipts": attempts,
        "terminal_accounting_consistent": values[2]+values[3] == attempts["terminal_attempts"],
        "launch_workload_exercised": values[1] > 0,
        "observer_actual_samples": observer.actual_samples,
        "fresh_imitation": fresh_imitation_health(env),
        "readiness": {"quiet_ticks": r.config.quiet_ticks, "maximum_settling_ticks": r.config.maximum_settling_ticks},
        "private_observer_memory": observer.metadata}


def fresh_imitation_health_matches(health: dict, expected_windows: int) -> bool:
    """Fail closed on missing, sanitized, skipped or duplicated WBC outputs."""
    from .runtime.imitation_rewards import UPSTREAM_NAMES
    if not isinstance(health, dict) or type(expected_windows) is not int or expected_windows < 0:
        return False
    counts, calls = health.get("term_nonfinite_counts"), health.get("term_call_counts")
    return (health.get("configured_terms") == list(UPSTREAM_NAMES)
        and type(health.get("observed_windows")) is int
        and health["observed_windows"] == expected_windows
        and health.get("all_raw_terms_finite") is True and health.get("every_term_each_window") is True
        and isinstance(counts, dict) and set(counts) == set(UPSTREAM_NAMES)
        and isinstance(calls, dict) and set(calls) == set(UPSTREAM_NAMES)
        and all(type(counts[name]) is int and counts[name] == 0
                and type(calls[name]) is int and calls[name] == expected_windows for name in UPSTREAM_NAMES))


def attach_engineering_cycle(env, admission, *, output: Path, measurement_contract: dict):
    """Attach before the first reset; shared by capacity and scratch engineering.

    Returns (bridge, bounded_sink). The caller must flush the sink at every
    phase boundary and in its exception/finalization path. Feedforward remains
    exactly the bundle's zero term; guard-learning checkpoints are not inputs.
    """
    from .engineering_admission import verify_engineering_admission
    from .telemetry.attempt_receipts import AggregateAttemptSink
    from .telemetry.cycle_runtime import attach_cycle_runtime
    from .telemetry.snapshot_providers import make_snapshot_providers
    identity = measurement_contract["payload"]["reference_and_body"]
    recipe = verify_recipe_manifest(identity.get("engineering_recipe"), identity.get("engineering_recipe_sha256"))
    require_engineering_configuration(env.cfg, engineering_recipe=recipe)
    bundle = load_runtime_bundle(env.command_manager.get_term("motion").cfg.bundle_path)
    verify_engineering_admission(bundle, admission)
    identity = measurement_contract["payload"]["reference_and_body"]
    if (measurement_contract["sha256"] != digest(measurement_contract["payload"])
            or identity.get("engineering_admission_sha256") != admission.sha256
            or identity.get("reference_manifest_sha256") != sha256_file(bundle.root/"manifest.json")
            or not identity.get("bridge_evidence_sha256") or not identity.get("observer_evidence_sha256")
            or (recipe.training.reference_start_probability > 0 and not identity.get("training_initialization_evidence_sha256"))):
        raise ValueError("Engineering cycle attachment requires the matching measured configuration/evidence contract")
    sink = AggregateAttemptSink(output, shared_identity={
        "measurement_contract_sha256": measurement_contract["sha256"],
        "reference_and_body": identity, "engineering_only": True,
        "acceptance_qualified": False, "feedforward_override": None}, flush_every_batches=128)
    bridge = attach_cycle_runtime(env, **make_snapshot_providers(env, engineering_admitted=True),
        receipt_sink=sink)
    return bridge, sink


def distributed_normalization_evidence(runner, *, include_parameters=False) -> dict:
    """Inspect actual rank-local buffers and collectively verify matching state."""
    import torch
    from .validation import tensor_digest
    models = {"actor": runner.alg.get_policy(), "critic": runner.alg._raw_critic}
    states = {}
    for name, model in models.items():
        buffers = {key: value for key, value in model.state_dict().items() if key.startswith("obs_normalizer.")}
        if not buffers or not all(torch.isfinite(value).all().item() for value in buffers.values()):
            raise ValueError("Actor/critic observation-normalization buffers are missing or nonfinite")
        states[name] = {"sha256": tensor_digest(buffers), "keys": list(buffers),
                       "count": int(buffers["obs_normalizer.count"].item())}
        if include_parameters:
            states[name]["model_state_sha256"] = tensor_digest(model.state_dict())
    gathered = [None]*runner.gpu_world_size
    if runner.is_distributed:
        torch.distributed.all_gather_object(gathered, states)
    else:
        gathered[0] = states
    hashes = [digest(state) for state in gathered]
    if len(set(hashes)) != 1:
        raise ValueError("Distributed actor/critic normalization buffers differ across ranks")
    return {"states": states, "rank_state_sha256": hashes, "all_ranks_match": True,
            "normalization_finite": True, "scope": "actual buffers at the recorded learner boundary"}


def initialization_health_matches(health: dict, engineering_recipe=LEGACY_RECIPE_ID) -> bool:
    """Require typed, disjoint strict/training counts, never relabel exposure."""
    recipe = resolve_engineering_recipe(engineering_recipe)
    if not isinstance(health, dict):
        return False
    names = ("cumulative_attempts", "cumulative_launches", "cumulative_strict_guard_starts",
        "cumulative_strict_ready_launches", "cumulative_training_reference_starts",
        "cumulative_training_support_grace_ticks")
    if (any(type(health.get(name)) is not int or health[name] < 0 for name in names)
            or health.get("training_initialization") != recipe.training.initialization_parameters()
            or health.get("launch_count_semantics") != "all_reference_starts_including_training_initialization"):
        return False
    attempts, launches, strict_starts, strict_launches, training_starts, grace = (health[name] for name in names)
    terminal = health.get("terminal_receipts", {})
    terminal_names = ("terminal_attempts", "launched_attempts", "strict_ready_launch_attempts",
        "training_reference_start_attempts", "unclassified_launch_attempts", "initialization_metadata_missing_attempts",
        "initial_support_grace_ticks_used")
    if not isinstance(terminal, dict) or any(type(terminal.get(name)) is not int or terminal[name] < 0 for name in terminal_names):
        return False
    return (attempts == strict_starts+training_starts and launches == strict_launches+training_starts
        and strict_launches <= strict_starts and terminal["terminal_attempts"] <= attempts
        and terminal["launched_attempts"] == terminal["strict_ready_launch_attempts"]+terminal["training_reference_start_attempts"]
        and terminal["strict_ready_launch_attempts"] <= strict_launches
        and terminal["training_reference_start_attempts"] <= training_starts
        and terminal["unclassified_launch_attempts"] == terminal["initialization_metadata_missing_attempts"] == 0
        and terminal["initial_support_grace_ticks_used"] <= grace
        and grace <= recipe.training.initial_support_grace_ticks*training_starts
        and (recipe.training.reference_start_probability > 0 or training_starts == grace == 0))


def _require_initialization_mode(report: dict, recipe) -> None:
    """Validate completed instrumentation; exposure never certifies readiness."""
    recipe = resolve_engineering_recipe(recipe)
    audit = report.get("initialization_audit", {})
    required = {"full_reference_arrays_unchanged", "strict_initialization_exercised",
        "training_initialization_exercised", "training_rows_start_at_frame_zero",
        "strict_rows_do_not_launch_at_reset", "reset_has_no_fabricated_readiness_dwell",
        "training_start_has_explicit_preparation", "first_observation_training_phase",
        "first_observation_strict_phase", "first_observation_has_zero_elapsed",
        "actor_future_reference_matches_command", "first_action_condition_unchanged_after_reset_commit",
        "first_action_reference_unchanged_after_reset_commit", "full_plan_identity_target_and_g1_context_frozen",
        "grace_exactly_matches_initial_unsupported_endpoints", "grace_has_four_tick_absolute_cap",
        "raw_support_flags_preserved", "all_receipts_explicitly_ineligible_and_start_typed"}
    inherited = report.get("inherited_native_audit", {})
    native_required = {"complete_native_samples", "complete_policy_samples", "ordinary_reset_exercised",
        "one_initialization_per_reset", "fresh_endpoint_fk", "fresh_actor_anchor",
        "finite_task_rewards", "finite_observations"}
    checks, native_checks = audit.get("checks", {}), inherited.get("checks", {})
    verify_recipe_manifest(report.get("recipe"), report.get("recipe_sha256"))
    if (report.get("recipe") != recipe.manifest() or report.get("recipe_sha256") != recipe.sha256
            or report.get("schema") != "human_wbc_training_initialization_native_v1"
            or report.get("status") != "passed" or report.get("exception") is not None
            or report.get("training") is not False or type(report.get("learning_updates")) is not int
            or report.get("learning_updates") != 0 or report.get("acceptance_qualified") is not False
            or report.get("reference_domain_qualified") is not False
            or report.get("runtime", {}).get("declared_pins_match") is not True
            or report.get("actor") != "zero_residual_no_network" or report.get("seed") != 10010
            or report.get("later_g1_metadata_perturbation") is not True
            or report.get("initialization") != recipe.training.initialization_parameters()
            or report.get("environments") != 16 or report.get("requested_policy_steps") != 400
            or report.get("completed_policy_steps") != 400 or report.get("actual_physics_samples") != 1600
            or report.get("observer_samples") != 1600 or report.get("actor_observation_dimension") != [398]
            or report.get("readiness") != {"quiet_ticks": 40, "maximum_settling_ticks": 1200}
            or report.get("stream_capture", {}).get("passed") is not True
            or not fresh_imitation_health_matches(report.get("fresh_imitation_health"), 400)
            or audit.get("status") != "passed" or not required <= checks.keys()
            or not all(value is True for value in checks.values())
            or inherited.get("status") != "passed" or not native_required <= native_checks.keys()
            or not all(value is True for value in native_checks.values())
            or inherited.get("launch_qualification") != "instrumentation_only_training_starts_separate"
            or audit.get("strict_launch_qualification") != "not_granted_by_this_instrumentation_audit"):
        raise ValueError("Completed matching native training initialization instrumentation is required")
    counts = [audit.get(name) for name in ("ordinary_reset_starts", "training_frame_zero_starts", "strict_guard_reset_starts")]
    if (any(type(value) is not int or value < 0 for value in counts) or counts[0] != counts[1]+counts[2]
            or counts[2] <= 0 or (recipe.training.reference_start_probability > 0 and counts[1] <= 0)
            or (recipe.training.reference_start_probability == 0 and counts[1] != 0)):
        raise ValueError("Native initialization report omits separately exercised strict/training starts")


def training_initialization_capacity_evidence(bundle, report_path: Path, *, engineering_recipe) -> dict:
    """Bind the current source, both native modes and every completion product."""
    recipe = resolve_engineering_recipe(engineering_recipe)
    if recipe.recipe_id == LEGACY_RECIPE_ID:
        raise ValueError("Legacy recipe must not borrow revised initialization evidence")
    record = _evidence(report_path)
    path = read_artifact(record)
    report = json.loads(path.read_text())
    completion_record = _evidence(path.parent/"completion_manifest.json")
    completion = json.loads(read_artifact(completion_record).read_text())
    packet_record = _evidence(path.parent.parent/"packet.json")
    execution_record = _evidence(path.parent.parent/"execution.json")
    packet = json.loads(read_artifact(packet_record).read_text())
    execution = json.loads(read_artifact(execution_record).read_text())
    modes = ("strict_default", "mixed_training")
    if (report.get("schema") != "human_wbc_training_initialization_panel_v1"
            or report.get("status") != "passed" or report.get("training") is not False
            or report.get("acceptance_qualified") is not False or report.get("strict_launch_qualified") is not False
            or completion.get("schema") != "human_wbc_training_initialization_completion_v1"
            or completion.get("status") != "passed" or completion.get("driver_returncode") != 0
            or completion.get("driver_exception") is not None or completion.get("acceptance_qualified") is not False
            or completion.get("strict_launch_qualified") is not False
            or completion.get("source_schema") != report["schema"]
            or completion.get("aggregate_report_sha256") != record["sha256"]
            or completion.get("completed_modes") != list(modes) or completion.get("learning_updates") != 0
            or execution.get("schema") != "human_wbc_training_initialization_execution_v1"
            or packet.get("schema") != "human_wbc_training_initialization_packet_v1"
            or packet.get("modes") != list(modes) or packet.get("training") is not False
            or packet.get("learning_updates") != 0 or packet.get("acceptance_qualified") is not False
            or packet.get("strict_launch_qualified") is not False
            or packet.get("environments_per_mode") != 16 or packet.get("policy_steps_per_mode") != 400
            or packet.get("physics_steps_per_mode") != 1600 or execution.get("learning_updates") != 0
            or execution.get("returncode") != 0 or execution.get("budget_stop") is not None
            or not execution.get("job_id") or execution.get("packet_sha256") != packet_record["sha256"]):
        raise ValueError("Native initialization panel completion/provenance is invalid")
    products = completion.get("products", {})
    if not isinstance(products, dict) or "report.json" not in products:
        raise ValueError("Native initialization completion products are missing")
    payloads = {}
    for relative, expected in products.items():
        product = (path.parent/relative).resolve(strict=True)
        if (not product.is_relative_to(path.parent.resolve()) or Path(relative).is_absolute()
                or (path.parent/relative).is_symlink()):
            raise ValueError("Native initialization product escapes its run directory")
        item = _evidence(product)
        if item["sha256"] != expected:
            raise ValueError("Native initialization completed product hash differs")
        payloads[relative] = item
    from .runtime.engineering_recipe import LEGACY_RECIPE
    checked = load_runtime_bundle(bundle)
    rows = report.get("modes", [])
    if [row.get("mode") for row in rows] != list(modes):
        raise ValueError("Native initialization requires exactly strict and revised modes")
    summaries = {}
    for name, selected, row in zip(modes, (LEGACY_RECIPE, recipe), rows, strict=True):
        key = name+"/report.json"
        mode_record = payloads.get(key)
        if (mode_record is None or row.get("status") != "passed"
                or row.get("report_sha256") != mode_record["sha256"]
                or completion.get("mode_report_sha256", {}).get(name) != mode_record["sha256"]):
            raise ValueError("Native initialization mode report identity differs")
        mode_report = json.loads(read_artifact(mode_record).read_text())
        _require_initialization_mode(mode_report, selected)
        configured = packet.get("configuration", {}).get(name, {})
        if configured.get("recipe") != selected.manifest() or configured.get("recipe_sha256") != selected.sha256:
            raise ValueError("Native initialization packet recipe differs from the executed mode")
        if mode_report.get("bundle_manifest_sha256") != sha256_file(checked.root/"manifest.json"):
            raise ValueError("Native initialization uses another reference/body/controller bundle")
        required = {"native", "pre_interval", "endpoint", "policy_boundary", "initial_observation"}
        if set(mode_report.get("hashes", {})) != required or name+"/compiled_scene.mjb" not in payloads:
            raise ValueError("Native initialization saved trace/model set is incomplete")
        for trace in required:
            if payloads.get(name+"/"+trace+".npz", {}).get("sha256") != mode_report["hashes"][trace]:
                raise ValueError("Native initialization raw trace identity differs")
        receipts = [key for key in payloads if key.startswith(name+"/attempts/") and key.endswith(".json")]
        if (type(mode_report.get("terminal_receipts")) is not int or mode_report["terminal_receipts"] <= 0
                or len(receipts) != mode_report["terminal_receipts"]):
            raise ValueError("Native initialization terminal receipt set is incomplete")
        mode_completion = completion.get("modes", {}).get(name, {})
        if (mode_completion.get("status") != "passed" or mode_completion.get("recipe_sha256") != selected.sha256
                or mode_completion.get("compiled_scene_sha256") != payloads[name+"/compiled_scene.mjb"]["sha256"]):
            raise ValueError("Native initialization compiled-model completion identity differs")
        summaries[name] = {"recipe_sha256": selected.sha256,
            "initialization_audit": mode_report["initialization_audit"],
            "native_instrumentation_status": mode_report["inherited_native_audit"]["status"]}
    runtime_files = {"action.py", "body.py", "bundle.py", "command.py", "mdp.py", "task.py",
        "task_rewards.py", "task_reward_adapter.py", "training_initialization.py", "engineering_recipe.py",
        "imitation_rewards.py"}
    sources = {relative: expected for relative, expected in packet["inputs"].items()
        if relative.startswith("src/human_punching/wbc/telemetry/")
        or (relative.startswith("src/human_punching/wbc/runtime/") and Path(relative).name in runtime_files)}
    required_sources = {"src/human_punching/wbc/"+relative for relative in (
        "runtime/training_initialization.py", "runtime/engineering_recipe.py", "runtime/task_rewards.py",
        "runtime/task_reward_adapter.py", "runtime/imitation_rewards.py", "telemetry/cycle_runtime.py",
        "telemetry/event_state.py", "telemetry/readiness_clock.py", "telemetry/attempt_receipts.py",
        "telemetry/training_initialization_qualification.py")}
    required_sources.update(OBSERVER_REQUIRED_SOURCES)
    required_sources.update("src/human_punching/wbc/runtime/"+name for name in runtime_files)
    if not required_sources <= sources.keys():
        raise ValueError("Native initialization source identity is incomplete")
    qualification = "src/human_punching/wbc/telemetry/training_initialization_qualification.py"
    if completion.get("qualification_source_sha256") != sources[qualification]:
        raise ValueError("Native initialization qualification source hash differs")
    for relative, expected in sources.items():
        read_artifact({"path": relative, "sha256": expected})
    return {"report": record, "completion_manifest": completion_record, "packet": packet_record,
        "execution": execution_record, "trace_payloads": payloads, "source_hashes": sources,
        "recipe": recipe.manifest(), "recipe_sha256": recipe.sha256, "modes": summaries,
        "instrumentation_health_passed": True, "acceptance_qualified": False,
        "strict_ready_launch_qualified": False, "production_throughput_measured": False}
