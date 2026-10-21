#!/usr/bin/env python3
"""Counterfactual isolated trace fixtures, not physical misses or moving-G1 dodges.

Read existing per-family historical rollouts from a Phase 1 packet. Change only
the diagnostic target supplied to two offline evaluators; never simulate or
change the stored trajectory, policy state, reference, or original result.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mujoco
import numpy as np

from human_punching.functional_recovery import (
    FunctionalGuardThresholds, derive_guard_nominal, evaluate_functional_trace,
    measure_functional_guard,
)
from human_punching.jab_evaluation import evaluate_rollout
from human_punching.wbc.provenance import sha256_file, writable_path, write_json
from human_punching.wbc.release_baseline_evaluation import evaluator_identity


def run_fixtures(source: Path, predicate: Path, output: Path) -> dict:
    source, predicate = source.resolve(strict=True), predicate.resolve(strict=True)
    output = writable_path(output)
    output.mkdir(parents=True, exist_ok=False)
    if not source.is_relative_to(ROOT):
        raise ValueError("Trace fixtures require preserved human-owned source packets")
    record = json.loads(predicate.read_text())
    thresholds = FunctionalGuardThresholds(**record["thresholds"])
    schedule = record["schedule"]
    identities = evaluator_identity()
    reports = []
    for family in ("jab", "cross", "left_hook", "right_hook"):
        candidates = sorted(source.rglob(f"{family}_accepted_ff_learned/historical_evaluation/rollout_00.npz"))
        # Native Phase 1 condition names may omit the residual label.
        if not candidates:
            candidates = sorted(source.rglob(f"{family}_accepted_ff/historical_evaluation/rollout_00.npz"))
        if len(candidates) != 1:
            raise ValueError(f"Exactly one nominal trial00 for {family} required; found {len(candidates)}")
        rollout = candidates[0]
        runtime = rollout.parent/"historical_evaluation_runtime.xml"
        reference = ROOT/"outputs/four-punch-milestone"/family/"reference.npz"
        inputs = {str(path): sha256_file(path) for path in (rollout, runtime, reference, predicate)}
        with np.load(rollout, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        with np.load(reference, allow_pickle=False) as archive:
            ref = {name: archive[name] for name in archive.files}
        model = mujoco.MjModel.from_xml_path(str(runtime))
        nominal = derive_guard_nominal(model, ref["qpos"][0], family=family,
            provenance=dict(reference_sha256=sha256_file(reference), fixture="counterfactual displaced target only"))
        # Fixed +10 m x displacement is declared before evaluation; no search
        # for a convenient target or changed trajectory is performed.
        original_target = np.asarray(ref["head_center_w"] if "head_center_w" in ref else ref["head_center"], float)
        displaced_target = original_target + [10., 0., 0.]
        measured = measure_functional_guard(model, arrays["qpos"], arrays["qvel"],
            arrays["support_contact"], nominal, displaced_target, thresholds=thresholds)
        events = schedule[family] if isinstance(schedule, dict) else next(row for row in schedule if row["family"] == family)
        revised, trace = evaluate_functional_trace(arrays["time_s"], measured,
            launch_time_s=events["launch_time_s"], outward_complete_time_s=events["outward_completion_time_s"],
            active_hand=0 if family in ("jab", "left_hook") else 1, thresholds=thresholds)
        historical = evaluate_rollout(model, arrays, ref, head_center=displaced_target,
                                      required_physics_timestep_s=.005, required_policy_timestep_s=.020)
        unchanged = all(sha256_file(path) == expected for path, expected in inputs.items())
        checks = dict(source_files_unchanged=unchanged,
                      revised_recovered_miss=bool(revised["functional_complete"] and revised["contact_outcome"] == "miss"),
                      no_any_hand_contact=not bool(np.any(measured["frozen_target_glove_distance_m"] <= .14)),
                      historical_contact_dependent_recovery_false=not historical["acceptance_checks"]["guard_recovered_with_recorded_support"],
                      historical_contact_false=not historical["any_glove_contact"],
                      initial_complete_false=revised["initial_completion_false"])
        folder = output/family
        folder.mkdir()
        with (folder/"measurements.npz").open("xb") as stream:
            np.savez_compressed(stream, **measured, **trace)
        write_json(folder/"revised.json", revised)
        write_json(folder/"historical.json", historical)
        result = dict(family=family, checks=checks, passed=all(checks.values()), source_sha256=inputs,
                      original_target_w=original_target.tolist(), diagnostic_target_w=displaced_target.tolist(),
                      recovery_time_s=revised["recovery_time_s"], contact_outcome=revised["contact_outcome"],
                      commanded_hit_success=False, nominal_complete_cycle=False)
        write_json(folder/"fixture.json", result)
        reports.append(result)
    summary = dict(schema="task10b_phase1_displaced_target_trace_fixture_v1", passed=all(row["passed"] for row in reports),
        fixture_count=len(reports), physical_trials_executed=0, actor_calls=0, physics_steps=0,
        moving_g1_qualification=False, physical_miss_qualification=False, weights_changed=False,
        limitation="Counterfactual isolated trace evaluation: diagnostic target displaced +10m x; the original physical trajectory hit its original target. This verifies contact-independent recovery software semantics only.",
        predicate=asdict(thresholds), evaluator_identity=identities, script_sha256=sha256_file(__file__), trials=reports)
    write_json(output/"summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "predicate", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    args = parser.parse_args()
    result = run_fixtures(args.source, args.predicate, args.output)
    print(json.dumps({"passed": result["passed"], "fixture_count": result["fixture_count"]}))
    raise SystemExit(0 if result["passed"] else 1)
