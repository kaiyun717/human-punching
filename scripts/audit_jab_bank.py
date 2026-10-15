#!/usr/bin/env python3
"""Independently audit every saved trial against one immutable evaluation bank."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from human_punching.jab_evaluation import evaluate_rollout


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def result_ranges(results):
    """Compact comparison evidence; missing contacts are counted, never zero-filled."""
    values = {
        "sole_drift_max_m": [max(row["sole_drift_max_m"]) for row in results],
        "guard_recovered_dwell_s": [row["final_guard"]["trailing_recovered_dwell_s"] for row in results],
        "joint_limit_violation_max_rad": [row["joint_limit_violation_max_rad"] for row in results],
        "actual_joint_torque_max_nm": [row["actual_joint_torque_max_nm"] for row in results],
        "active_glove_position_rms_m": [(row["reference_tracking"] or {}).get("active_glove_position_rms_m") for row in results],
        "attack_active_glove_position_rms_m": [(row["reference_tracking"] or {}).get("attack_active_glove_position_rms_m") for row in results],
        "signed_interpolated_contact_error_s": [(row["contact_timing"] or {}).get("signed_interpolated_contact_error_s") for row in results],
        "absolute_interpolated_contact_error_s": [(row["contact_timing"] or {}).get("absolute_interpolated_contact_error_s") for row in results],
    }
    ranges = {}
    for name, samples in values.items():
        finite = [value for value in samples if value is not None]
        ranges[name] = {"minimum": min(finite) if finite else None,
                        "maximum": max(finite) if finite else None,
                        "mean": float(np.mean(finite)) if finite else None,
                        "measured_trials": len(finite), "missing_trials": len(samples) - len(finite)}
    return ranges


def audit_trial(arguments):
    model, reference, path, output, attack_window, trial_id, physics_dt, policy_dt = arguments
    metrics = evaluate_rollout(model, path, reference, output,
                               attack_window_reference_s=attack_window,
                               required_physics_timestep_s=physics_dt,
                               required_policy_timestep_s=policy_dt)
    return {
        "trial_id": trial_id, "rollout": str(path), "evaluation": str(output),
        "pilot_pass": metrics["pilot_pass"],
        "acceptance_checks": metrics["acceptance_checks"],
        "failed_checks": metrics["failed_checks"],
        "any_cycle_contact": metrics["any_glove_contact"],
        "attack_window_contact": metrics["attack_window_glove_contact"],
        "attack_minimum_glove_distance_m": metrics["attack_window_glove_min_distance_m"],
        "root_height_min_m": metrics["root_height_min_m"],
        "root_tilt_max_rad": metrics["root_tilt_max_rad"],
        "sole_drift_max_m": metrics["sole_position_drift_max_m"],
        "final_guard": metrics["final_guard"],
        "reference_tracking": metrics["reference_tracking"],
        "contact_timing": metrics["contact_timing"],
        "precision_targets": metrics["precision_targets"],
        "punch_family": metrics["punch_family"],
        "active_hand": metrics["active_hand"],
        "effective_source_speed": metrics["effective_source_speed"],
        "reference_timing": metrics["reference_timing"],
        "physics_timestep_s": metrics["evaluated_physics_timestep_s"],
        "policy_timestep_s": metrics["recorded_policy_timestep_s"],
        "actuator_control_kind": metrics["actuator_control_kind"],
        "actual_joint_torque_max_nm": metrics["actual_joint_torque_max_nm"],
        "joint_limit_violation_max_rad": metrics["joint_limit_violation_max_rad"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runtime-model", type=Path, help="Frozen compiled-controller XML; --model remains the bank/source model for hash provenance")
    parser.add_argument("--checkpoint", type=Path, help="Require every recorded checkpoint hash to match this frozen policy")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=Path("outputs/scalar-evaluation-bank/guard_trials.npz"))
    parser.add_argument("--bank-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--require-physics-timestep", type=float)
    parser.add_argument("--require-policy-timestep", type=float)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    with np.load(args.bank, allow_pickle=False) as source:
        bank = {key: source[key] for key in source.files}
    manifest = json.loads((args.bank_manifest or args.bank.parent / "manifest.json").read_text())
    attack_window = manifest["attack_window_reference_s"]
    bank_hash = digest(args.bank)
    if digest(args.model) != str(bank["model_sha256"].item()):
        raise ValueError("Model differs from evaluation bank model")
    if digest(args.reference) != str(bank["reference_sha256"].item()):
        raise ValueError("Reference differs from evaluation bank reference")
    rows = []
    for path in sorted(args.rollouts.glob("rollout_*.npz")):
        with np.load(path, allow_pickle=False) as rollout:
            field = next((key for key in ("trial_id", "world_id", "evaluation_bank_world") if key in rollout), "trial_id")
            if field not in rollout or "evaluation_bank_sha256" not in rollout:
                raise ValueError(f"{path}: requires trial/world ID and evaluation_bank_sha256")
            trial_id = int(rollout[field].item())
            if trial_id not in bank["trial_ids"]:
                raise ValueError(f"{path}: trial ID not in supplied bank")
            if str(rollout["evaluation_bank_sha256"].item()) != bank_hash:
                raise ValueError(f"{path}: evaluation-bank hash mismatch")
            for provenance in ("model_sha256", "reference_sha256"):
                if provenance not in rollout or str(rollout[provenance].item()) != str(bank[provenance].item()):
                    raise ValueError(f"{path}: recorded {provenance} differs from bank")
            if args.runtime_model is not None:
                if "runtime_model_sha256" not in rollout or str(rollout["runtime_model_sha256"].item()) != digest(args.runtime_model):
                    raise ValueError(f"{path}: runtime model hash differs from saved compiled controller")
            if args.checkpoint is not None:
                if "checkpoint_sha256" not in rollout or str(rollout["checkpoint_sha256"].item()) != digest(args.checkpoint):
                    raise ValueError(f"{path}: checkpoint hash differs from the supplied policy")
            bank_row = int(np.flatnonzero(bank["trial_ids"] == trial_id)[0])
            for key in ("qpos", "qvel"):
                if not np.allclose(rollout[key][0], bank[key][bank_row], atol=3e-7, rtol=0):
                    raise ValueError(f"{path}: initial {key} differs from bank trial {trial_id}")
        rows.append((trial_id, path))
    if sorted(row[0] for row in rows) != sorted(bank["trial_ids"].tolist()):
        raise ValueError("Exactly one recorded rollout for every bank trial is required")
    args.output.mkdir(parents=True, exist_ok=True)
    arguments = [(args.runtime_model or args.model, args.reference, path, args.output / f"trial_{trial_id:02d}.json",
                  attack_window, trial_id, args.require_physics_timestep,
                  args.require_policy_timestep) for trial_id, path in rows]
    if args.workers == 1:
        results = [audit_trial(row) for row in arguments]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(audit_trial, arguments))
    count = len(results)
    pass_count = sum(row["pilot_pass"] for row in results)
    checks = {key: sum(row["acceptance_checks"][key] for row in results)
              for key in results[0]["acceptance_checks"]}
    summary = {
        "bank": str(args.bank), "bank_sha256": bank_hash,
        "model_sha256": str(bank["model_sha256"].item()),
        "reference_sha256": str(bank["reference_sha256"].item()),
        "runtime_model_sha256": digest(args.runtime_model) if args.runtime_model is not None else None,
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "verified_checkpoint_sha256": digest(args.checkpoint) if args.checkpoint is not None else None,
        "required_physics_timestep_s": args.require_physics_timestep,
        "required_policy_timestep_s": args.require_policy_timestep,
        "trial_count": count, "all_initial_states_match_bank": True,
        "attack_window_reference_s": attack_window,
        "pilot_pass_count": pass_count, "pilot_pass_fraction": pass_count / count,
        "gate_pass_counts": checks,
        "gate_pass_fractions": {key: value / count for key, value in checks.items()},
        "any_cycle_contact_count": sum(row["any_cycle_contact"] for row in results),
        "precision_target_pass_counts": {key: sum(row["precision_targets"][key] for row in results)
                                         for key in ("active_glove_rms_target_met", "contact_timing_target_met")},
        "ranges": result_ranges(results),
        "nominal_trial": next(row for row in results if row["trial_id"] == 0),
        "trials": results,
        "scope": "Paired small initial-guard perturbations of one annotated punch reference; not held-out motion or target generalization",
        "statistics_note": f"Descriptive counts on a fixed {count}-trial bank; no population success guarantee",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("trials", "nominal_trial")}, indent=2))


if __name__ == "__main__":
    main()
