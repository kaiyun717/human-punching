#!/usr/bin/env python3
"""Validate and atomically package four independently accepted 5 ms punch policies.

Run with --check-only before publication. No output directory is created unless
every family passes preflight. This packages existing evidence; it does not run
physics, select the best checkpoint, relax gates, or certify generalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
FAMILIES = ("jab", "cross", "left_hook", "right_hook")
HANDS = {"jab": "Left", "cross": "Right", "left_hook": "Left", "right_hook": "Right"}
REQUIRED_GATES = frozenset({
    "one_guard_reset_and_every_physics_step_recorded", "starts_from_reference_guard",
    "no_fall", "sole_drift", "joint_limits", "bounded_recorded_control",
    "bounded_recorded_joint_torque", "no_external_forces", "unactuated_root",
    "attack_window_contact", "guard_recovered_with_recorded_support",
    "reference_glove_tracking", "recorded_actuator_force_matches_joint_torque",
    "recorded_controller_matches_compiled_actuator", "recorded_runtime_model_hash_matches",
    "active_hand_attack_contact", "required_physics_timestep", "required_policy_update_interval",
})
PRECISION_TARGETS = ("active_glove_rms_target_met", "contact_timing_target_met")
BASELINE_MATCH_GATES = REQUIRED_GATES - {
    "no_fall", "sole_drift", "joint_limits", "attack_window_contact",
    "guard_recovered_with_recorded_support", "reference_glove_tracking", "active_hand_attack_contact",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(value, expected):
    return value is not None and np.isfinite(value) and abs(float(value)-expected) < 1e-9


def resolve(path, project=PROJECT):
    path = Path(path)
    return (path if path.is_absolute() else project / path).resolve()


def load_checkpoint(path):
    # Our checkpoints contain tensors and basic containers only. Do not allow
    # arbitrary Python globals while inspecting a user-selected artifact.
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def precision(row):
    targets = row["precision_targets"]
    tracking = row.get("reference_tracking") or {}
    timing = row.get("contact_timing") or {}
    full = tracking.get("active_glove_position_rms_m")
    attack = tracking.get("attack_active_glove_position_rms_m")
    error = timing.get("absolute_interpolated_contact_error_s")
    return {
        "full_cycle_active_glove_rms_2cm": bool(full is not None and np.isfinite(full) and full <= .02
                                                and targets["active_glove_rms_target_met"]),
        "attack_active_glove_rms_2cm": bool(attack is not None and np.isfinite(attack) and attack <= .02),
        "contact_timing_20ms": bool(error is not None and np.isfinite(error) and error <= .02
                                    and targets["contact_timing_target_met"]),
    }


def reference_timing_metadata(reference, allow_conditioned_timing=False):
    kind = str(np.asarray(reference.get("timing_kind", "uniform_source_playback")).item())
    modified = "nonuniform" in kind or bool(np.asarray(reference.get("source_timing_modified", False)).item())
    if modified:
        require(allow_conditioned_timing, "Nonuniform source timing requires explicit --allow-conditioned-timing")
        low, high = (float(reference[name]) for name in ("local_source_speed_min", "local_source_speed_max"))
        require(np.isfinite([low, high]).all() and 0 < low <= high, "Invalid declared local source-speed range")
        return {"timing_kind": kind, "source_timing_modified": True, "effective_source_speed": None,
                "local_source_speed_min": low, "local_source_speed_max": high,
                "description": str(np.asarray(reference["timing_description"]).item())}
    require("source_speed" in reference and close(float(reference["source_speed"]), 1.),
            "Uniform reference must use source-speed 1.0x")
    return {"timing_kind": kind, "source_timing_modified": False, "effective_source_speed": 1.}


def validate_family(family, checkpoint_path, evaluation_dir, *, project=PROJECT,
                    require_precision=False, allow_conditioned_timing=False,
                    checkpoint_loader=load_checkpoint):
    """Return a reviewed file-copy plan; never write or mutate source artifacts."""
    require(family in FAMILIES, f"Unknown family: {family}")
    checkpoint_path = resolve(checkpoint_path, project)
    evaluation_dir = resolve(evaluation_dir, project)
    training_dir = checkpoint_path.parent
    independent = evaluation_dir / "independent"
    summary = read_json(independent / "summary.json")
    saved = checkpoint_loader(checkpoint_path)
    training_schema = read_json(training_dir / "controller_schema.json")
    evaluation_schema = read_json(evaluation_dir / "controller_schema.json")
    require(saved["controller_schema"] == training_schema == evaluation_schema,
            f"{family}: training/checkpoint/evaluation controller schemas differ")
    require(saved.get("policy") and saved.get("actions", 0) > 0,
            f"{family}: checkpoint has no learned policy")
    schema = training_schema
    require(close(schema.get("physics_dt"), .005) and close(schema.get("actor_dt"), .02),
            f"{family}: checkpoint must use native 5 ms physics and 20 ms actions")
    require(close(schema.get("speed"), 1.), f"{family}: runtime playback must be 1.0x")
    require(schema.get("controller_mode") == "implicit_pd", f"{family}: expected validated implicit PD controller")
    require(schema.get("refresh_kinematics") is True, f"{family}: expected current-state kinematics")
    require(schema.get("balance_feedback") is False, f"{family}: unexpected auxiliary balance feedback")
    pairs = {
        "model_sha256": (training_dir / "model/human.xml", evaluation_dir / "model/human.xml"),
        "reference_sha256": (training_dir / "reference.npz", evaluation_dir / "reference.npz"),
        "runtime_model_sha256": (training_dir / "model/human_runtime.xml", evaluation_dir / "model/human_runtime.xml"),
    }
    hashes = {}
    for key, (trained, evaluated) in pairs.items():
        hashes[key] = sha256(trained)
        require(sha256(evaluated) == hashes[key] == summary.get(key),
                f"{family}: {key} differs across training/evaluation/audit")
        if key != "runtime_model_sha256":
            require(saved.get(key) == schema.get(key) == hashes[key], f"{family}: checkpoint {key} mismatch")
    runtime = ET.parse(pairs["runtime_model_sha256"][0]).getroot()
    option = runtime.find("option")
    require(option is not None and close(float(option.get("timestep", "nan")), .005),
            f"{family}: saved runtime XML does not use 5 ms")
    require(sha256(training_dir / "body_metadata.json") == sha256(evaluation_dir / "body_metadata.json"),
            f"{family}: training/evaluation body metadata differs")
    with np.load(training_dir / "reference.npz", allow_pickle=False) as reference:
        require(str(reference["punch_family"].item()) == family, f"{family}: wrong reference family")
        require(str(reference["active_hand"].item()) == HANDS[family], f"{family}: wrong reference active hand")
        timing_metadata = reference_timing_metadata(reference, allow_conditioned_timing)
        attack_window = reference["attack_window_reference_s"].tolist()
    trial_count = summary.get("trial_count", 0)
    require(isinstance(trial_count, int) and trial_count >= 32
            and summary.get("pilot_pass_count") == trial_count,
            f"{family}: require at least 32 independent trials and every trial to pass")
    require(summary.get("all_initial_states_match_bank") is True, f"{family}: unmatched initial-state bank")
    require(close(summary.get("required_physics_timestep_s"), .005)
            and close(summary.get("required_policy_timestep_s"), .02), f"{family}: audit did not require native clocks")
    counts = summary["gate_pass_counts"]
    require(REQUIRED_GATES <= counts.keys() and all(value == trial_count for value in counts.values()),
            f"{family}: required acceptance gate missing or failed")
    require(np.allclose(summary["attack_window_reference_s"], attack_window, atol=1e-10, rtol=0),
            f"{family}: audit attack window differs from annotated reference")
    bank_path = evaluation_dir / "evaluation_bank.npz"
    bank_hash = sha256(bank_path)
    require(bank_hash == summary["bank_sha256"], f"{family}: evaluation bank hash mismatch")
    checkpoint_hash = sha256(checkpoint_path)
    if summary.get("verified_checkpoint_sha256") is not None:
        require(summary["verified_checkpoint_sha256"] == checkpoint_hash,
                f"{family}: audit checkpoint hash differs from selected checkpoint")
    files = {}

    def add(source, destination):
        source = Path(source)
        require(source.is_file(), f"{family}: missing artifact {source}")
        files[destination] = {"source": source, "sha256": sha256(source)}

    with np.load(bank_path, allow_pickle=False) as bank:
        ids = [int(value) for value in bank["trial_ids"]]
        require(len(ids) == trial_count and len(set(ids)) == trial_count and 0 in ids,
                f"{family}: invalid complete evaluation bank")
        for key in ("model_sha256", "reference_sha256"):
            require(str(bank[key].item()) == hashes[key], f"{family}: bank {key} mismatch")
        trials = summary["trials"]
        require(sorted(int(row["trial_id"]) for row in trials) == sorted(ids), f"{family}: duplicate/missing audited trial")
        precision_counts = dict.fromkeys(("full_cycle_active_glove_rms_2cm", "attack_active_glove_rms_2cm", "contact_timing_20ms"), 0)
        checkpoint_hash_recorded = 0
        for row in trials:
            trial_id = int(row["trial_id"])
            detailed_path = independent / f"trial_{trial_id:02d}.json"
            detailed = read_json(detailed_path)
            for item in (row, detailed):
                checks = item["acceptance_checks"]
                require(REQUIRED_GATES <= checks.keys() and all(value is True for value in checks.values())
                        and item["pilot_pass"] is True and item["failed_checks"] == [],
                        f"{family}: trial {trial_id} is not independently accepted")
                require(item["punch_family"] == family and item["active_hand"] == HANDS[family],
                        f"{family}: trial {trial_id} family/hand mismatch")
                if timing_metadata["source_timing_modified"]:
                    trial_timing = item.get("reference_timing") or {}
                    timing_kind = trial_timing.get("timing_kind", item.get("timing_kind"))
                    require(item.get("effective_source_speed") is None
                            and timing_kind == timing_metadata["timing_kind"],
                            f"{family}: trial {trial_id} mislabels nonuniform timing")
                    for key in ("local_source_speed_min", "local_source_speed_max"):
                        if key in trial_timing:
                            require(close(trial_timing[key], timing_metadata[key]),
                                    f"{family}: trial {trial_id} mislabels nonuniform timing rate")
                else:
                    require(close(item["effective_source_speed"], 1.), f"{family}: trial {trial_id} is not 1.0x")
            require(row["acceptance_checks"] == detailed["acceptance_checks"], f"{family}: inconsistent trial audit")
            require(close(detailed["evaluated_physics_timestep_s"], .005)
                    and close(detailed["model_xml_timestep_s"], .005)
                    and close(detailed["recorded_policy_timestep_s"], .02), f"{family}: detailed trial clocks mismatch")
            require(detailed["simulator_clock_evidence"]["verified"] is True,
                    f"{family}: missing actual simulator-clock evidence")
            achieved = precision(detailed)
            require(achieved == precision(row), f"{family}: inconsistent precision audit")
            for key, value in achieved.items():
                precision_counts[key] += int(value)
            if require_precision:
                require(all(achieved.values()), f"{family}: trial {trial_id} misses a precision target")
            rollout_path = resolve(row["rollout"], project)
            require(rollout_path.parent == evaluation_dir, f"{family}: rollout is outside supplied evaluation run")
            with np.load(rollout_path, allow_pickle=False) as recorded:
                require(int(recorded["trial_id"].item()) == trial_id, f"{family}: rollout trial ID mismatch")
                for key, value in {**hashes, "evaluation_bank_sha256": bank_hash}.items():
                    require(str(recorded[key].item()) == value, f"{family}: rollout {key} mismatch")
                recorded_checkpoint = resolve(str(recorded["checkpoint"].item()), project)
                require(recorded_checkpoint == checkpoint_path, f"{family}: evaluation used a different checkpoint path")
                if "checkpoint_sha256" in recorded:
                    require(str(recorded["checkpoint_sha256"].item()) == checkpoint_hash,
                            f"{family}: evaluated checkpoint bytes differ")
                    checkpoint_hash_recorded += 1
                else:
                    require(re.fullmatch(r"checkpoint_\d+\.pt", checkpoint_path.name),
                            f"{family}: mutable checkpoint filename lacks recorded SHA256")
                require(str(recorded["evaluation_kind"].item()) == "learned_policy", f"{family}: rollout is not learned")
                require(close(float(recorded["physics_timestep_s"]), .005)
                        and close(float(recorded["policy_timestep_s"]), .02)
                        and close(float(recorded["reference_speed"]), 1.), f"{family}: recorded clocks mismatch")
                times = recorded["time_s"]
                require(np.allclose(np.diff(times), .005, atol=1e-8, rtol=1e-5)
                        and int(recorded["state_reset_count"]) == 1
                        and int(recorded["physics_step_count"]) == len(times)-1,
                        f"{family}: incomplete physics-step recording")
                require(np.array_equal(recorded["policy_update_step_indices"], np.arange(0, len(times)-1, 4)),
                        f"{family}: policy update indices do not establish 20 ms actions")
                bank_row = ids.index(trial_id)
                for key in ("qpos", "qvel"):
                    require(np.allclose(recorded[key][0], bank[key][bank_row], atol=3e-7, rtol=0),
                            f"{family}: rollout initial {key} differs from bank")
            add(rollout_path, f"evaluation/{rollout_path.name}")
            add(detailed_path, f"evaluation/independent/{detailed_path.name}")

    for relative in ("model/human.xml", "model/human_runtime.xml", "body_metadata.json",
                     "reference.npz", "controller_schema.json", "arguments.json",
                     "jab_learning_source.py", "scalar_control_source.py"):
        add(training_dir / relative, relative)
    add(checkpoint_path, "checkpoint.pt")
    add(bank_path, "evaluation/evaluation_bank.npz")
    add(independent / "summary.json", "evaluation/independent/summary.json")
    if summary.get("bank"):
        bank_manifest = resolve(summary["bank"], project).parent / "manifest.json"
        bank_metadata = read_json(bank_manifest)
        require(bank_metadata["bank_sha256"] == bank_hash,
                f"{family}: original bank manifest hash mismatch")
        add(bank_manifest, "evaluation/evaluation_bank_manifest.json")
        reference_origin = resolve(bank_metadata["reference"], project)
        require(sha256(reference_origin) == hashes["reference_sha256"],
                f"{family}: original reference changed after bank creation")
        for name, destination in (("manifest.json", "reference_manifest.json"),
                                  ("scalar_reference_source.py", "scalar_reference_source.py"),
                                  ("retime_punch_reference_source.py", "retime_punch_reference_source.py")):
            if (reference_origin.parent / name).is_file():
                add(reference_origin.parent / name, destination)
    for relative in ("arguments.json", "controller_schema.json", "jab_learning_source.py",
                     "scalar_control_source.py", "metrics.json"):
        add(evaluation_dir / relative, f"evaluation/{relative}")
    for relative in ("training.jsonl", "learning-curve.png", "learning-curve.svg", "initialization.json"):
        if (training_dir / relative).is_file():
            add(training_dir / relative, relative)
    record = {
        "family": family, "active_hand": HANDS[family], "training_directory": str(training_dir),
        "selected_checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_update": int(saved["update"]), "evaluation_directory": str(evaluation_dir),
        "trial_count": trial_count, "accepted_physical_pilot_trials": trial_count,
        "gate_pass_counts": counts, "precision_target_pass_counts": precision_counts,
        "all_precision_targets_met": all(value == trial_count for value in precision_counts.values()),
        "precision_targets_required_for_this_release": require_precision,
        "checkpoint_hash_recorded_trials": checkpoint_hash_recorded,
        "checkpoint_provenance_limitation": None if checkpoint_hash_recorded == trial_count else
            "Legacy numbered checkpoints are matched by path and current hash; missing historical byte hashes are not reconstructed.",
        "physics_timestep_s": .005, "policy_timestep_s": .02, "reference_playback_speed": 1.,
        "source_speed": timing_metadata["effective_source_speed"], "reference_timing": timing_metadata,
        "conditioned_timing_explicitly_allowed": allow_conditioned_timing,
        "attack_window_reference_s": attack_window, "hashes": hashes,
        "ranges": summary.get("ranges", {}),
        "scope": summary.get("scope", "Small fixed initial-state perturbation bank"),
    }
    return {"record": record, "files": files}


def attach_optional_evidence(plan, *, media_dir=None, baseline_dir=None, project=PROJECT):
    """Attach explicitly selected supporting artifacts after their provenance checks."""
    record, files = plan["record"], plan["files"]
    family = record["family"]
    learned = resolve(record["evaluation_directory"], project)
    optional = record.setdefault("optional_evidence", {})

    def add(source, relative):
        source = Path(source)
        require(source.is_file(), f"{family}: missing supporting artifact {source}")
        require(relative not in files, f"{family}: duplicate supporting destination {relative}")
        files[relative] = {"source": source, "sha256": sha256(source)}

    if media_dir is not None:
        media = resolve(media_dir, project)
        require(media.is_dir(), f"{family}: missing media directory")
        namespace = f"{media.parent.parent.name}-{media.name}" if media.parent.name == "independent" else media.name
        render = media / "render.json"
        media_run = learned
        render_kind = None
        if render.is_file():
            evidence = read_json(media / "evaluation.json")
            render_kind = evidence.get("evaluation_kind")
            require(evidence.get("punch_family") == family, f"{family}: rendered evidence has a different punch family")
            if render_kind == "reference_pd":
                require("zero_residual_baseline" in optional,
                        f"{family}: attach the paired baseline before its rendered evidence")
                media_run = resolve(optional["zero_residual_baseline"]["directory"], project)
                require(resolve(evidence["model_xml"], project) == media_run / "model/human_runtime.xml",
                        f"{family}: rendered baseline belongs to a different run")
            else:
                require(render_kind == "learned_policy"
                        and evidence.get("recorded_checkpoint_sha256") == record["checkpoint_sha256"],
                        f"{family}: rendered evidence does not match the selected learned checkpoint")
        validation_path = media / "video-validation.json"
        validation = read_json(validation_path) if validation_path.is_file() else {}
        if validation.get("recorded_rollout"):
            source_rollout = resolve(validation["recorded_rollout"], project)
            artifact_group = "baseline" if render_kind == "reference_pd" else "evaluation"
            require(source_rollout.parent == media_run
                    and f"{artifact_group}/{source_rollout.name}" in files,
                    f"{family}: rendered video uses a different evaluation rollout")
        selected = [path for path in sorted(media.rglob("*")) if path.is_file()]
        require(selected, f"{family}: empty media directory")
        videos = [path for path in selected if path.suffix.lower() in {".mp4", ".webm", ".mov"}]
        if videos:
            require(len(videos) == 1 and validation_path.is_file(),
                    f"{family}: each video requires its matching full-decode validation")
            require(validation.get("recorded_rollout"),
                    f"{family}: video validation must identify its recorded rollout")
            decode_code = validation.get("full_decode_exit_code", validation.get("ffmpeg_returncode"))
            require(decode_code == 0 and resolve(validation["video"], project) == videos[0].resolve(),
                    f"{family}: video decoding failed or validation refers to a different video")
            if "bytes" in validation:
                require(validation["bytes"] == videos[0].stat().st_size,
                        f"{family}: video size changed after its decode validation")
        for path in selected:
            require(path.resolve().is_relative_to(media), f"{family}: media symlink escapes supplied directory")
            add(path, f"media/{namespace}/{path.relative_to(media).as_posix()}")
        optional.setdefault("media", []).append({"directory": str(media), "artifact_count": len(selected),
            "package_directory": f"media/{namespace}",
            "render_kind": render_kind,
            "render_checkpoint_verified": render_kind == "learned_policy",
            "render_baseline_verified": render_kind == "reference_pd",
            "video_full_decode_verified": bool(videos),
            "scope": "Supporting visual evidence; acceptance remains based on the recorded physics trials."})

    if baseline_dir is not None:
        baseline = resolve(baseline_dir, project)
        summary = read_json(baseline / "independent/summary.json")
        learned_summary = read_json(learned / "independent/summary.json")
        require(read_json(baseline / "controller_schema.json") == read_json(learned / "controller_schema.json"),
                f"{family}: zero-residual baseline controller schema differs")
        require(read_json(baseline / "arguments.json").get("checkpoint") is None
                and read_json(baseline / "metrics.json").get("kind") == "reference_pd",
                f"{family}: supplied baseline is not the zero-residual reference controller")
        count = record["trial_count"]
        require(summary.get("trial_count") == count
                and summary.get("all_initial_states_match_bank") is True,
                f"{family}: baseline trial bank is incomplete or mismatched")
        bank_hash = sha256(learned / "evaluation_bank.npz")
        require(sha256(baseline / "evaluation_bank.npz") == summary["bank_sha256"] == bank_hash,
                f"{family}: baseline and learned initial-state banks differ")
        for key, relative in (("model_sha256", "model/human.xml"),
                              ("runtime_model_sha256", "model/human_runtime.xml"),
                              ("reference_sha256", "reference.npz")):
            require(sha256(baseline / relative) == summary.get(key) == record["hashes"][key],
                    f"{family}: baseline {key} differs")
        require(sha256(baseline / "body_metadata.json") == sha256(learned / "body_metadata.json"),
                f"{family}: baseline body metadata differs")
        require(close(summary.get("required_physics_timestep_s"), .005)
                and close(summary.get("required_policy_timestep_s"), .02),
                f"{family}: baseline audit clocks differ")
        rows = summary["trials"]
        learned_ids = sorted(int(row["trial_id"]) for row in learned_summary["trials"])
        require(sorted(int(row["trial_id"]) for row in rows) == learned_ids,
                f"{family}: baseline audited trials do not pair with learned trials")
        require(summary["pilot_pass_count"] == sum(row["pilot_pass"] is True for row in rows),
                f"{family}: inconsistent baseline success count")
        with np.load(baseline / "evaluation_bank.npz", allow_pickle=False) as bank:
            ids = list(bank["trial_ids"])
            for row in rows:
                trial_id = int(row["trial_id"])
                detailed_path = baseline / f"independent/trial_{trial_id:02d}.json"
                detailed = read_json(detailed_path)
                checks = row["acceptance_checks"]
                require(checks == detailed["acceptance_checks"]
                        and BASELINE_MATCH_GATES <= checks.keys()
                        and all(checks[key] is True for key in BASELINE_MATCH_GATES),
                        f"{family}: baseline changed the execution/actuation contract")
                require(detailed.get("simulator_clock_evidence", {}).get("verified") is True,
                        f"{family}: baseline lacks simulator-clock evidence")
                rollout = resolve(row["rollout"], project)
                require(rollout.parent == baseline, f"{family}: baseline rollout is outside supplied run")
                with np.load(rollout, allow_pickle=False) as saved:
                    require(str(saved["evaluation_kind"].item()) == "reference_pd"
                            and int(saved["trial_id"]) == trial_id,
                            f"{family}: baseline rollout is not the paired zero-residual trial")
                    for key, value in {**record["hashes"], "evaluation_bank_sha256": bank_hash}.items():
                        require(str(saved[key].item()) == value, f"{family}: baseline rollout {key} mismatch")
                    require(close(float(saved["physics_timestep_s"]), .005)
                            and close(float(saved["policy_timestep_s"]), .02)
                            and close(float(saved["reference_speed"]), 1.),
                            f"{family}: baseline rollout clocks differ")
                    times = saved["time_s"]
                    require(np.allclose(np.diff(times), .005, atol=1e-8, rtol=1e-5)
                            and int(saved["state_reset_count"]) == 1
                            and int(saved["physics_step_count"]) == len(times)-1
                            and np.array_equal(saved["policy_update_step_indices"], np.arange(0, len(times)-1, 4)),
                            f"{family}: baseline rollout lacks every-step/20ms-action evidence")
                    for key in ("qpos", "qvel"):
                        require(np.allclose(saved[key][0], bank[key][ids.index(trial_id)], atol=3e-7, rtol=0),
                                f"{family}: baseline initial {key} differs from the paired bank")
                add(rollout, f"baseline/{rollout.name}")
                add(detailed_path, f"baseline/independent/{detailed_path.name}")
        for relative in ("independent/summary.json", "evaluation_bank.npz", "model/human.xml",
                         "model/human_runtime.xml", "reference.npz", "body_metadata.json",
                         "controller_schema.json", "arguments.json", "metrics.json",
                         "jab_learning_source.py", "scalar_control_source.py"):
            add(baseline / relative, f"baseline/{relative}")
        paired_path = resolve(learned_summary["bank"], project).parent / "paired-results.json" if learned_summary.get("bank") else None
        if paired_path is not None and paired_path.is_file():
            paired = read_json(paired_path)
            require(resolve(paired["learned_run"], project) == learned
                    and resolve(paired["zero_residual_baseline_run"], project) == baseline
                    and paired["bank_sha256"] == bank_hash
                    and paired["controller_schemas_identical"] is True
                    and paired["trials"] == count
                    and paired["learned_pilot_pass_count"] == record["accepted_physical_pilot_trials"]
                    and paired["baseline_pilot_pass_count"] == summary["pilot_pass_count"],
                    f"{family}: saved paired comparison belongs to different runs")
            add(paired_path, "baseline/paired-results.json")
        optional["zero_residual_baseline"] = {
            "directory": str(baseline), "trial_count": count, "bank_sha256": bank_hash,
            "controller_schemas_identical": True,
            "learned_pilot_pass_count": record["accepted_physical_pilot_trials"],
            "baseline_pilot_pass_count": summary["pilot_pass_count"],
            "baseline_gate_pass_counts": summary["gate_pass_counts"],
            "scope": "Same frozen controller, models, reference, native clocks and initial states with residual disabled. Baseline is not required to pass physical success gates and is not an optimized classical controller."}
    return plan


def supporting_directory_files(directories, *, project=PROJECT):
    files = {}
    for directory in directories:
        directory = resolve(directory, project)
        require(directory.is_dir(), f"Missing supporting evidence directory: {directory}")
        for source in sorted(directory.rglob("*")):
            if not source.is_file():
                continue
            require(source.resolve().is_relative_to(directory), "Supporting evidence symlink escapes its directory")
            relative = f"supporting/{directory.name}/{source.relative_to(directory).as_posix()}"
            require(relative not in files, f"Duplicate supporting artifact: {relative}")
            files[relative] = {"source": source, "sha256": sha256(source)}
    return files


def package_release(plans, output, *, project=PROJECT, supporting_files=None):
    """Publish only after all explicit preflight plans have passed; refuse overwrite."""
    require(set(plans) == set(FAMILIES), "A release requires exactly all four punch families")
    output, project = resolve(output, project), project.resolve()
    require(output.is_relative_to(project / "outputs"), "Release must remain inside human-punching/outputs")
    require(not output.exists(), "Release destination already exists; use a new versioned directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    artifacts = {}
    try:
        for family in FAMILIES:
            for relative, item in plans[family]["files"].items():
                destination = staging / family / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item["source"], destination)
                require(sha256(destination) == item["sha256"], f"Artifact changed since validation: {item['source']}")
                artifacts[f"{family}/{relative}"] = {"sha256": item["sha256"],
                    "bytes": destination.stat().st_size, "origin": str(item["source"])}
        for relative, item in (supporting_files or {}).items():
            require(relative.startswith("supporting/") and ".." not in Path(relative).parts,
                    "Invalid supporting evidence destination")
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item["source"], destination)
            require(sha256(destination) == item["sha256"], f"Supporting artifact changed since validation: {item['source']}")
            artifacts[relative] = {"sha256": item["sha256"], "bytes": destination.stat().st_size,
                                   "origin": str(item["source"])}
        shared = {"package_punch_policies_source.py": Path(__file__).resolve()}
        for name in ("package_four_punch_milestone.sh", "package_four_punch_milestone.sbatch",
                     "train_punch_5ms_volta.sbatch"):
            shared[f"reproduction/{name}"] = Path(__file__).with_name(name).resolve()
        for name in ("LICENSE.md", "NOTICE.md", "THIRD_PARTY_NOTICES.txt", "manifest.json"):
            shared[f"licenses/protomotions/{name}"] = project / "assets/vendor/protomotions" / name
        for relative, source in shared.items():
            require(source.is_file(), f"Missing release provenance artifact: {source}")
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            artifacts[relative] = {"sha256": sha256(destination), "bytes": destination.stat().st_size,
                                   "origin": str(source)}
        precision_required = all(plans[family]["record"]["precision_targets_required_for_this_release"]
                                 for family in FAMILIES)
        manifest = {
            "schema": "four_punch_physical_policy_release_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "acceptance_level": ("independently accepted physical pilots with all declared precision targets required and passed"
                                 if precision_required else
                                 "independently accepted physical pilots; precision targets reported separately"),
            "precision_targets_required_for_all_families": precision_required,
            "families": {family: plans[family]["record"] for family in FAMILIES},
            "artifacts": artifacts,
            "limitations": ["Four separate reference-conditioned policies, not a mixed-punch scheduler",
                "One conditioned source crop and fixed target per family; no moving-G1 or impact validation",
                "Small initial-guard perturbations; no held-out motion/person/target generalization claim",
                "Attack windows are geometry-derived, not reviewed capture outward-phase annotations"],
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        (staging / "README.md").write_text(
            "# Four-punch 5 ms physical policy release\n\n"
            "Each family has its own checkpoint, frozen source/runtime XML, reference, controller schema, "
            "training source and complete independent evaluation evidence (at least 32 trials per family). Read `manifest.json` "
            "for hashes, artifact origins, acceptance and precision-target counts. Physics uses 5 ms; "
            "policy updates use 20 ms; frozen references replay at 1.0×. Original-source timing "
            "and any explicitly allowed nonuniform timing are reported separately for each family.\n\n"
            "These are four isolated physical pilots. They do not provide requested punch rates, "
            "continuous combinations, moving-target generalization or a complete adversarial opponent. "
            "Historical source paths inside copied evidence are retained for provenance.\n\n"
            "## Check bundled artifact hashes\n\n"
            "Run from the `human-punching` project root:\n\n"
            "```bash\n.venv/bin/python -B - <<'PY'\n"
            "import hashlib, json\nfrom pathlib import Path\n"
            f"bundle = Path({str(output.relative_to(project))!r})\n"
            "manifest = json.loads((bundle / 'manifest.json').read_text())\n"
            "for relative, metadata in manifest['artifacts'].items():\n"
            "    assert hashlib.sha256((bundle / relative).read_bytes()).hexdigest() == metadata['sha256'], relative\n"
            "print('All bundled artifact hashes match')\nPY\n```\n\n"
            "## Evaluate a bundled policy on the GPU\n\n"
            "Run from the same project root with the existing project environments. This submits a new evaluation "
            "using the bundled model, reference, checkpoint and frozen 64-trial bank. The runner restores the "
            "checkpoint's controller settings and requires 5 ms physics / 20 ms actions. Change `punch_family` "
            "to `cross`, `left_hook` or `right_hook` for another family. Results go to a new job directory; "
            "the release remains unchanged.\n\n"
            "```bash\npunch_family=jab\n"
            f"bundle_root={str(output.relative_to(project))}\n"
            'family_bundle="$bundle_root/$punch_family"\n'
            'sbatch "$bundle_root/reproduction/train_punch_5ms_volta.sbatch" \\\n'
            '  --mode evaluate --model "$family_bundle/model/human.xml" \\\n'
            '  --body-metadata "$family_bundle/body_metadata.json" \\\n'
            '  --reference "$family_bundle/reference.npz" \\\n'
            '  --checkpoint "$family_bundle/checkpoint.pt" \\\n'
            '  --evaluation-bank "$family_bundle/evaluation/evaluation_bank.npz" \\\n'
            '  --recorded-worlds 64\n```\n\n'
            "The bundled Slurm wrapper uses the existing repository runner and installed dependencies. "
            "Exact training/evaluation source snapshots are preserved per family for provenance; this bundle "
            "does not vendor the complete software environment. Original selection/publication commands are "
            "preserved under `reproduction/`; their historical input paths are not needed by the bundled-policy "
            "evaluation command above.\n")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return output


def mapping(values, *, require_all=True, allow_multiple=False):
    result = {}
    for value in values:
        family, separator, path = value.partition("=")
        require(separator and family in FAMILIES and path and (allow_multiple or family not in result),
                "Use exactly one FAMILY=PATH entry for each known family")
        if allow_multiple:
            require(Path(path) not in result.get(family, []), "Duplicate supporting directory")
            result.setdefault(family, []).append(Path(path))
        else:
            result[family] = Path(path)
    if require_all:
        require(set(result) == set(FAMILIES), "Provide explicit paths for all four families")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="FAMILY=PATH")
    parser.add_argument("--evaluation", action="append", required=True, metavar="FAMILY=RUN_DIR")
    parser.add_argument("--media-dir", action="append", default=[], metavar="FAMILY=DIR",
                        help="Optional videos/images/plots copied with provenance; repeat for multiple directories per family")
    parser.add_argument("--baseline", action="append", default=[], metavar="FAMILY=RUN_DIR",
                        help="Optional paired zero-residual evaluation; controller and initial-state bank must match")
    parser.add_argument("--support-dir", action="append", default=[], type=Path,
                        help="Optional global evidence directory copied with hashes under supporting/")
    parser.add_argument("--output", type=Path, default=Path("outputs/four-punch-milestone"))
    parser.add_argument("--require-precision", action="store_true",
                        help="Additionally require full-cycle/attack RMS <=2 cm and timing error <=20 ms in all trials")
    parser.add_argument("--check-only", action="store_true", help="Validate selected families without creating output; publication requires all four")
    parser.add_argument("--allow-conditioned-timing", action="store_true",
                        help="Explicitly admit declared nonuniformly retimed references; never relabel them uniform source-speed 1x")
    args = parser.parse_args()
    checkpoints = mapping(args.checkpoint, require_all=not args.check_only)
    evaluations = mapping(args.evaluation, require_all=not args.check_only)
    require(checkpoints.keys() == evaluations.keys(), "Checkpoint and evaluation families must match")
    media = mapping(args.media_dir, require_all=False, allow_multiple=True)
    baselines = mapping(args.baseline, require_all=False)
    require(media.keys() <= checkpoints.keys() and baselines.keys() <= checkpoints.keys(),
            "Supporting evidence must belong to a selected checkpoint family")
    plans = {family: validate_family(family, checkpoints[family], evaluations[family],
                                     require_precision=args.require_precision,
                                     allow_conditioned_timing=args.allow_conditioned_timing)
             for family in FAMILIES if family in checkpoints}
    for family, plan in plans.items():
        attach_optional_evidence(plan, baseline_dir=baselines.get(family))
        for media_dir in media.get(family, []):
            attach_optional_evidence(plan, media_dir=media_dir)
    supporting_files = supporting_directory_files(args.support_dir)
    print(json.dumps({family: plan["record"] for family, plan in plans.items()}, indent=2))
    if not args.check_only:
        print(f"Published {package_release(plans, args.output, supporting_files=supporting_files)}")


if __name__ == "__main__":
    main()
