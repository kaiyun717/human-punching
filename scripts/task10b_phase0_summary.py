#!/usr/bin/env python3
"""Summarize completed Task 10B Phase 0 evidence; no simulator import or execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("jab", "cross", "left_hook", "right_hook")
SOLVERS = ("historical", "accepted")
METRICS = {
    "full_cycle_glove_rms_m": ("reference_tracking", "active_glove_position_rms_m"),
    "outward_glove_rms_m": ("reference_tracking", "attack_active_glove_position_rms_m"),
    "historical_contact_error_s": ("contact_timing", "absolute_interpolated_contact_error_s"),
    "guard_dwell_s": ("final_guard", "trailing_recovered_dwell_s"),
    "joint_limit_excursion_rad": ("joint_limit_violation_max_rad",),
    "maximum_joint_torque_nm": ("actual_joint_torque_max_nm",),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path, identities):
    path = Path(path).resolve(strict=True)
    identities[str(path)] = sha256(path)
    return json.loads(path.read_text())


def checked_child(root, relative):
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Evidence path escapes its packet: " + str(relative))
    return path


def read_passed_packet(path, phase, identities):
    path = Path(path).resolve(strict=True)
    if path.is_dir():
        path /= "packet.json"
    packet = read_json(path, identities)
    receipt = read_json(path.parent/"execution.json", identities)
    modes = ["contract", "diagnostic"] if phase == "preflight" else ["parity"]
    if (packet.get("task") != "10B" or packet.get("program_phase") != 0
            or packet.get("phase") != phase or packet.get("training") is not False
            or receipt.get("status") != "passed" or receipt.get("completed_modes") != modes
            or receipt.get("packet_sha256") != sha256(path)):
        raise ValueError("A complete passed Phase 0 packet is required: " + str(path))
    if len(receipt.get("modes", [])) != len(modes):
        raise ValueError("Mode receipt count differs")
    audits = {}
    for mode, entry in zip(modes, receipt["modes"], strict=True):
        audit_path = checked_child(path.parent, f"run/{mode}/audit.json")
        audit = read_json(audit_path, identities)
        if (entry.get("mode") != mode or entry.get("returncode") != 0
                or entry.get("audit_status") != "passed" or entry.get("audit_sha256") != sha256(audit_path)
                or audit.get("status") != "passed" or audit.get("mode") != mode
                or audit.get("unattempted_conditions") != 0):
            raise ValueError("Passed audit receipt identity differs: " + mode)
        audits[mode] = audit
    return path.parent, packet, audits


def indexed_trials(rows, count):
    result = {row["trial_id"]: row for row in rows}
    if len(result) != len(rows) or sorted(result) != list(range(count)):
        raise ValueError(f"Expected exactly matched trial IDs 0..{count-1}")
    return result


def numeric_metric(row, path):
    value = row
    for key in path:
        value = (value or {}).get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError("Nonfinite or nonnumeric saved metric: " + "/".join(path))
    return float(value)


def distribution(values):
    finite = [v for v in values if v is not None]
    return dict(requested=len(values), observed=len(finite), missing=len(values)-len(finite),
                minimum=min(finite) if finite else None,
                median=statistics.median(finite) if finite else None,
                mean=statistics.fmean(finite) if finite else None,
                maximum=max(finite) if finite else None)


def pair_panel_metrics(historical, accepted):
    before, after = indexed_trials(historical, 64), indexed_trials(accepted, 64)
    result = {}
    for name, path in METRICS.items():
        a = [numeric_metric(before[i], path) for i in range(64)]
        b = [numeric_metric(after[i], path) for i in range(64)]
        delta = [None if x is None or y is None else y-x for x, y in zip(a, b, strict=True)]
        result[name] = dict(trial_ids=list(range(64)), historical=a, accepted=b,
            accepted_minus_historical=delta, historical_summary=distribution(a),
            accepted_summary=distribution(b), paired_difference_summary=distribution(delta))
    return result


def count_gates(rows):
    keys = set(rows[0]["acceptance_checks"])
    if any(set(row["acceptance_checks"]) != keys for row in rows):
        raise ValueError("Historical acceptance gate schema differs between trials")
    return {key: sum(row["acceptance_checks"][key] for row in rows) for key in sorted(keys)}


def collect(preflight, parity):
    identities = {}
    pre_root, pre_packet, pre = read_passed_packet(preflight, "preflight", identities)
    par_root, par_packet, par = read_passed_packet(parity, "parity", identities)
    if (pre_packet["files"] != par_packet["files"]
            or par_packet.get("preflight_packet_sha256") != sha256(pre_root/"packet.json")
            or par_packet.get("preflight_sha256") != sha256(pre_root/"execution.json")):
        raise ValueError("Parity does not bind this preflight and identical scientific inputs")
    diagnostic, panels, paired, caps = [], {}, {}, {}
    expected_diag = {(f, s, ff) for f in FAMILIES for s, ff in
                     (("historical", True), ("accepted", True), ("accepted", False))}
    seen = set()
    for condition in pre["diagnostic"]["conditions"]:
        key = (condition["family"], condition["solver_profile"], condition["feedforward_enabled"])
        if key not in expected_diag or key in seen or condition["status"] != "passed":
            raise ValueError("Missing, duplicate or failed diagnostic condition")
        seen.add(key)
        source = pre["diagnostic"]["source_inputs"][key[0]]
        if (condition["initial_bank_sha256"] != source["bank"]
                or condition["checkpoint_sha256"] != source["checkpoint"]):
            raise ValueError("Diagnostic bank/checkpoint identity differs")
        rows = list(indexed_trials(condition["diagnostics"], 16).values())
        diagnostic.append(dict(condition=condition["name"], family=key[0], solver=key[1],
            feedforward=key[2], attempts=16,
            contacts=sum(row["analytic_contact"] for row in rows), falls=sum(row["fell"] for row in rows),
            no_active_contact=sum(not row["analytic_contact"] for row in rows),
            final_one_second_guard=sum(row["final_one_second_guard"] for row in rows),
            final_guard_shorter_than_one_second=sum(not row["final_one_second_guard"] for row in rows),
            wrong_hand_contacts=sum(row["wrong_hand_analytic_contact"] for row in rows),
            forbidden_collision_diagnostic=sum(row["forbidden_collision"] for row in rows),
            actual_physics_steps=condition["actual_physics_steps"],
            status="instrumentation passed; physical results are separate",
            source_raw_trace_sha256=condition["raw_trace_sha256"], trials=rows))
    if seen != expected_diag:
        raise ValueError("Twelve complete diagnostic conditions are required")
    for condition in par["parity"]["conditions"]:
        key = (condition["family"], condition["solver_profile"], condition["residual_mode"])
        if (key in panels or key[0] not in FAMILIES or key[1] not in SOLVERS
                or key[2] not in ("learned", "zero") or condition["status"] != "passed"
                or condition["feedforward_enabled"] is not True
                or not all(condition["audit"]["checks"].values())):
            raise ValueError("Invalid or duplicate historical panel")
        summary_path = checked_child(par_root, f"run/parity/{condition['name']}/historical_evaluation/summary.json")
        summary = read_json(summary_path, identities)
        if summary != condition["historical_evaluation"]:
            raise ValueError("Historical summary differs from receipt-bound native audit")
        rows = list(indexed_trials(summary["trials"], 64).values())
        gate_counts = count_gates(rows)
        precision_counts = {name:sum(row["precision"][name] for row in rows) for name in rows[0]["precision"]}
        if (summary["worlds"] != 64 or summary["trial_count"] != 64
                or summary["strict_historical_release_pass_count"] != sum(r["strict_historical_release_pass"] for r in rows)
                or summary["pilot_pass_count"] != sum(r["pilot_pass"] for r in rows)
                or summary["gate_pass_counts"] != gate_counts
                or summary["precision_target_pass_counts"] != precision_counts
                or summary["historical_initial_bank_sha256"] != condition["initial_bank_sha256"]
                or summary["source_inputs"]["checkpoint"] != condition["checkpoint_sha256"]):
            raise ValueError("Historical count or input identity mismatch")
        panels[key] = dict(condition=condition["name"], family=key[0], solver=key[1], residual=key[2],
            attempts=64, strict_successes=summary["strict_historical_release_pass_count"],
            pilot_successes=summary["pilot_pass_count"],
            strict_failures=64-summary["strict_historical_release_pass_count"],
            gate_pass_counts=gate_counts,
            gate_failure_counts={k:64-v for k, v in gate_counts.items()},
            precision_target_pass_counts=precision_counts,
            precision_target_failure_counts={name:64-count for name,count in precision_counts.items()},
            source_inputs=summary["source_inputs"],
            historical_evaluation_duration_s=summary["historical_evaluation_duration_s"],
            actual_physics_steps=condition["actual_physics_steps"],
            excluded_terminal_padding_steps=summary["excluded_terminal_padding_steps"],
            trial_ids=list(range(64)), trials=rows)
    expected = {(f, s, r) for f in FAMILIES for s in SOLVERS for r in ("learned", "zero")}
    if set(panels) != expected:
        raise ValueError("Sixteen complete 64-world parity panels are required")
    for family in FAMILIES:
        reference = panels[family, "historical", "learned"]["source_inputs"]
        if any(panels[family, solver, residual]["source_inputs"] != reference
               for solver in SOLVERS for residual in ("learned", "zero")):
            raise ValueError("Paired bank, checkpoint or other source identities differ: " + family)
        if pre["diagnostic"]["source_inputs"][family] != reference:
            raise ValueError("Diagnostic and parity family source identities differ")
        schema_path = checked_child(par_root, f"source_snapshot/{par_packet['release']}/{family}/controller_schema.json")
        schema = read_json(schema_path, identities)
        if sha256(schema_path) != reference["schema"]:
            raise ValueError("Torque-cap schema identity mismatch")
        caps[family] = max(abs(v) for limits in schema["torque_limits"] for v in limits)
        paired[family] = pair_panel_metrics(panels[family, "historical", "learned"]["trials"],
                                           panels[family, "accepted", "learned"]["trials"])
    return dict(schema="task10b_phase0_saved_results_summary_v1", task="10B", phase=0,
        training=False, recomputed_physics=False, source_script_sha256=sha256(__file__),
        input_sha256=identities, diagnostic_conditions=diagnostic,
        parity_panels=[panels[key] for key in sorted(panels)], learned_solver_pairs=paired,
        largest_actuator_effort_cap_nm=caps,
        counts=dict(diagnostic_requested=192, diagnostic_recorded=192,
                    parity_requested=1024, parity_recorded=1024, unattempted=0),
        labels=dict(scope="Isolated historical fixed-reference physical execution; original guards and banks",
            timing="Error relative to original frozen-reference contact; not requested-TTC qualification",
            guard="Original exact-pose supported recovery, not revised functional recovery",
            missing_metrics="Retained as null; pair summaries disclose missing counts out of all64",
            exclusions="No G1 interaction, target/start expansion, scheduling, learning or promotion"))


def plots(summary, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    settings = [("full_cycle_glove_rms_m", "Full-cycle active glove RMS (m)", .02),
        ("outward_glove_rms_m", "Outward active glove RMS (m)", .02),
        ("historical_contact_error_s", "Historical-reference contact error (s)", .02),
        ("guard_dwell_s", "Historical exact-guard dwell (s)", .20),
        ("joint_limit_excursion_rad", "Joint-limit excursion (rad)", .02),
        ("maximum_joint_torque_nm", "Maximum joint torque (Nm)", None)]
    artifacts = {}
    for family in FAMILIES:
        fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
        for axis, (name, label, threshold) in zip(axes.flat, settings, strict=True):
            values = summary["learned_solver_pairs"][family][name]
            for solver, color in (("historical", "#2459a6"), ("accepted", "#c94f27")):
                ys = [float("nan") if v is None else v for v in values[solver]]
                axis.plot(range(64), ys, ".", markersize=5, color=color,
                          label=("Historical solver" if solver == "historical" else "Accepted G1 solver")
                          +f" ({values[solver+'_summary']['observed']}/64 observed)")
            if threshold is None:
                threshold = summary["largest_actuator_effort_cap_nm"][family]
                label_line = "Largest actuator cap; per-joint caps audited separately"
            else:
                label_line = "Historical minimum dwell" if name == "guard_dwell_s" else "Historical upper target/limit"
            axis.axhline(threshold, color="#555555", linestyle="--", linewidth=1, label=label_line)
            axis.set(xlabel="Matched original bank trial ID", ylabel=label, xlim=(-1, 64))
            axis.grid(alpha=.2)
            axis.legend(fontsize=7, loc="best")
        fig.suptitle(f"Task 10B Phase 0 — {family.replace('_', ' ')} — 64 learned-policy trials per solver\n"
                     "Historical fixed references and guards; contact timing is not a requested-TTC qualification", fontsize=13)
        target = destination/f"{family}_learned_solver_metrics.png"
        fig.savefig(target, dpi=150)
        plt.close(fig)
        artifacts[target.name] = sha256(target)
    return artifacts


def markdown(summary):
    lines = ["# Task 10B Phase 0 — saved-result summary", "",
        "All counts use saved reports from passed native packets. No physics or FK was recomputed. "
        "These are isolated historical fixed-reference results, not requested-TTC, moving-G1, or functional-guard qualification.", "",
        "## Twelve controlled diagnostic conditions", "",
        "Each denominator is 16. Guard denotes the original diagnostic's final uninterrupted one-second quiet guard.", "",
        "| Condition | Contact | Fall | Final guard | Wrong hand |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for row in summary["diagnostic_conditions"]:
        lines.append(f"| {row['condition']} | {row['contacts']}/16 | {row['falls']}/16 | {row['final_one_second_guard']}/16 | {row['wrong_hand_contacts']}/16 |")
    lines += ["", "## Original 64-state parity panels", "",
        "Strict success adds full/outward active-glove RMS and historical-reference contact-timing requirements to the original pilot gates. "
        "Zero-residual controls and all physical failures remain in the denominator.", "",
        "| Family | Solver | Residual | Strict | Pilot | Failed gates (count) | Failed precision targets (count) |",
        "| --- | --- | --- | ---: | ---: | --- | --- |"]
    for row in summary["parity_panels"]:
        failures = "; ".join(f"{k}: {v}" for k,v in row["gate_failure_counts"].items() if v) or "none"
        precision = "; ".join(f"{k}: {v}" for k,v in row["precision_target_failure_counts"].items() if v) or "none"
        lines.append(f"| {row['family']} | {row['solver']} | {row['residual']} | {row['strict_successes']}/64 | {row['pilot_successes']}/64 | {failures} | {precision} |")
    lines += ["", "## Matched learned-policy solver comparison", "",
        "Changes are accepted-G1-solver minus historical-solver for the same 64 original trial IDs. "
        "Bank, reference, checkpoint, controller schema, and other family input hashes match. "
        "Both panels still use the isolated human scene; no G1 body is present.", "",
        "| Family | Metric | Historical median | Accepted median | Paired mean change | Observed pairs |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    def number(value):
        return "missing" if value is None else f"{value:.6g}"
    for family, metrics in summary["learned_solver_pairs"].items():
        for name, value in metrics.items():
            lines.append(f"| {family} | {name} | {number(value['historical_summary']['median'])} | "
                f"{number(value['accepted_summary']['median'])} | {number(value['paired_difference_summary']['mean'])} | "
                f"{value['paired_difference_summary']['observed']}/64 |")
    lines += ["", "Guard dwell uses the historical 0.20 s criterion. The diagnostic one-second guard count is a separate measurement. "
        "Contact-error targets refer to the original fixed reference; no commanded TTC range has been qualified. "
        "Parity excludes final actor-rounding padding from historical evaluation while retaining it in native raw evidence.", "",
        "## Figures and machine-readable evidence", "",
        "The JSON ledger retains all 1,024 parity rows, all 192 diagnostic rows, every gate count, per-trial paired values, "
        "missing-metric denominators, input hashes, and the generating script identity.", "",
        "[Count, failure, and paired-metric ledger](summary.json)", ""]
    for family in FAMILIES:
        lines.append(f"- [{family.replace('_', ' ').title()} — all 64 paired trials]({family}_learned_solver_metrics.png)")
    return "\n".join(lines)+"\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    destination = args.output.resolve()
    if not destination.is_relative_to((ROOT/"outputs/task10b").resolve()):
        raise ValueError("Summary outputs must stay under human-punching/outputs/task10b")
    summary = collect(args.preflight, args.parity)
    destination.mkdir(parents=True, exist_ok=False)
    summary["plot_sha256"] = plots(summary, destination)
    (destination/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)+"\n")
    (destination/"README.md").write_text(markdown(summary))
    print(json.dumps(dict(output=str(destination), counts=summary["counts"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
