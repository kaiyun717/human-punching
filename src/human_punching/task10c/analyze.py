"""Compact plots and denominator-preserving tables from actual Phase 2 outputs.

This reads completed JSONL rows and evaluation summaries only. It neither runs
simulation nor chooses a policy. Re-running updates derived analysis artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .targets import PROJECT_ROOT


def _read(path: Path, snapshots: dict) -> bytes:
    data = path.read_bytes()
    snapshots[str(path)] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    return data


def read_learning(path: Path, snapshots: dict) -> tuple[list[dict], int]:
    """An active writer may leave a partial final line; never skip interior errors."""
    lines = _read(path, snapshots).splitlines(keepends=True)
    rows, partial_bytes = [], 0
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise
            partial_bytes = len(line)
    iterations = [row["completed_iterations"] for row in rows]
    if any(b <= a for a, b in zip(iterations, iterations[1:])):
        raise ValueError("Learning iterations are not strictly increasing")
    return rows, partial_bytes


def development_table(paths: list[Path], snapshots: dict) -> list[dict]:
    rows = []
    for path in paths:
        summary = json.loads(_read(path, snapshots))
        recovery = summary["recovery"]["all"]
        whole_body = summary.get("whole_body", {})
        row = {
            "iteration": int(path.parent.name.removeprefix("update_")),
            "status": summary["status"], "requested": summary["requested_attempts"],
            "attempted": summary["attempted"], "completed": summary["completed"],
            "hits": summary["hit_count"], "hit_denominator": summary["hit_denominator"],
            "misses": summary["miss_count"], "falls": summary["fall_count"],
            "fall_denominator": summary["fall_denominator"],
            "hand_pelvis_settled_terminal": recovery["settled_at_terminal"],
            "recovery_denominator": recovery["denominator"],
            "completed_without_fall": recovery["complete_horizon_without_fall"],
            "timing_denominator": summary["timing_error_denominator"],
            "signed_error_mean_ms": _milliseconds(summary["signed_timing_error_s"]["mean"]),
            "signed_error_p05_ms": _milliseconds(summary["signed_timing_error_s"]["p05"]),
            "signed_error_p95_ms": _milliseconds(summary["signed_timing_error_s"]["p95"]),
            "absolute_error_mean_ms": _milliseconds(summary["absolute_timing_error_s"]["mean"]),
            "early_below_0_5_s": summary["early_encounters_below_0_5_s"],
            "incidental_other_glove": summary["incidental_other_glove_encounters"],
            "selection_score": summary["selection_score"],
            "recipe_sha256": summary["recipe_sha256"],
            "requested_ttc_min_s": summary["requested_ttc_s"]["min"],
            "requested_ttc_max_s": summary["requested_ttc_s"]["max"],
            "terminal_guard_error_mean_m": recovery["terminal_guard_error_m"]["mean"],
            "terminal_guard_speed_mean_m_s": recovery["terminal_guard_speed_m_s"]["mean"],
            "whole_body_denominator": whole_body.get("denominator", 0),
            "terminal_whole_body_score_mean": whole_body.get("terminal_tracking_score", {}).get("mean"),
            "terminal_whole_body_score_p05": whole_body.get("terminal_tracking_score", {}).get("p05"),
            "terminal_hold_support_quality_mean": whole_body.get("terminal_hold_support_quality", {}).get("mean"),
            "terminal_hold_support_quality_p05": whole_body.get("terminal_hold_support_quality", {}).get("p05"),
            "maximum_command_excess_mean_rad": whole_body.get("maximum_command_excess_rad", {}).get("mean"),
            "maximum_command_excess_p95_rad": whole_body.get("maximum_command_excess_rad", {}).get("p95"),
            "maximum_command_excess_max_rad": whole_body.get("maximum_command_excess_rad", {}).get("max"),
            "raw_components": {key: value["mean"] for key, value in summary["raw_component_totals"].items()},
            "summary_path": str(path),
        }
        for side in ("left", "right"):
            loads = whole_body.get("terminal_hold_foot_vertical_force_mean_N", {}).get(side, {})
            durations = whole_body.get("terminal_hold_foot_positive_vertical_load_duration_s", {}).get(side, {})
            row[f"terminal_hold_{side}_vertical_load_mean_N"] = loads.get("mean")
            row[f"terminal_hold_{side}_vertical_load_denominator"] = loads.get("count", 0)
            row[f"terminal_hold_{side}_positive_load_duration_mean_s"] = durations.get("mean")
        for body, distribution in whole_body.get("terminal_body_tilt_degrees", {}).items():
            row[f"terminal_{body}_tilt_mean_degrees"] = distribution["mean"]
        rows.append(row)
    return sorted(rows, key=lambda row: row["iteration"])


def _milliseconds(value):
    return None if value is None else 1000 * value


def _ratio(numerator, denominator):
    return float(numerator) / denominator if denominator else float("nan")


def _rolling_component(learning: list[dict], name: str, window: int):
    totals = np.array([row["episodes"]["component_sums"].get(name, 0.) for row in learning])
    counts = np.array([row["episodes"]["episodes"] if name in row["episodes"]["component_sums"] else 0
                       for row in learning])
    kernel = np.ones(min(window, len(learning)))
    numerator = np.convolve(totals, kernel, mode="full")[:len(learning)]
    denominator = np.convolve(counts, kernel, mode="full")[:len(learning)]
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def analyze(run: str | Path, output: str | Path | None = None, rolling_window=25) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(run).resolve()
    run.relative_to(PROJECT_ROOT / "outputs/task10c/jab")
    output = Path(output).resolve() if output else run / "analysis"
    output.relative_to(PROJECT_ROOT / "outputs/task10c/jab")
    output.mkdir(parents=True, exist_ok=True)
    snapshots = {}
    learning, partial_bytes = read_learning(run / "learning.jsonl", snapshots)
    if not learning:
        raise ValueError("No completed learning rows are available")
    development = development_table(sorted((run / "development").glob("update_*/summary.json")), snapshots)
    iterations = np.array([row["completed_iterations"] for row in learning])
    transitions = np.array([row["total_transitions"] for row in learning])
    elapsed = np.array([row["elapsed_seconds"] for row in learning]) / 60
    dev_it = np.array([row["iteration"] for row in development])
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    observed_components = {name for row in learning for name in row["episodes"]["component_sums"]}
    observed_components.update(name for row in development for name in row["raw_components"])
    preferred_order = ("encounter", "interaction", "anchor_height", "anchor_orientation",
                       "body_position", "body_orientation", "body_linear_velocity", "body_angular_velocity",
                       "style", "balance", "guard", "command_excess", "action_rate", "effort", "joint_margin")
    component_names = [name for name in preferred_order if name in observed_components]
    component_names += sorted(observed_components - set(preferred_order))
    palette = plt.get_cmap("tab20")
    colors = {name: palette(index % 20) for index, name in enumerate(component_names)}
    axes[0, 0].plot(elapsed, transitions / 1e6, color="#1f77b4")
    axes[0, 0].set(xlabel="Application elapsed minutes", ylabel="Cumulative transitions (millions)",
                   title="Actual training exposure")
    for name, color in colors.items():
        axes[0, 1].plot(iterations, _rolling_component(learning, name, rolling_window),
                        label=name, color=color, linewidth=1.4)
    axes[0, 1].set(xlabel="Completed PPO updates", ylabel="Raw return / logged episode",
                   title=f"Training full + snippet mixture ({rolling_window}-update window)")
    axes[0, 1].legend(fontsize=7, ncol=2)
    for name, color in colors.items():
        axes[0, 2].plot(dev_it, [row["raw_components"].get(name, np.nan) for row in development],
                        marker="o", markersize=4, label=name, color=color)
    axes[0, 2].set(xlabel="Completed PPO updates", ylabel="Mean raw return / full attempt",
                   title="Deterministic guard-start development components")
    axes[0, 2].legend(fontsize=7, ncol=2)
    for label, key, denominator, color in (
        ("hit", "hits", "hit_denominator", "#1f77b4"),
        ("fall", "falls", "fall_denominator", "#d62728"),
        ("hand/pelvis settled (limited)", "hand_pelvis_settled_terminal", "recovery_denominator", "#2ca02c"),
        ("full horizon without fall", "completed_without_fall", "recovery_denominator", "#9467bd"),
    ):
        axes[1, 0].plot(dev_it, [_ratio(row[key], row[denominator]) for row in development],
                        marker="o", markersize=4, label=label, color=color)
    has_whole_body = any(row["whole_body_denominator"] for row in development)
    if has_whole_body:
        for key, label, color in (
            ("terminal_whole_body_score_mean", "whole-body tracking score", "#8c564b"),
            ("terminal_hold_support_quality_mean", "hold support quality", "#17becf"),
        ):
            axes[1, 0].plot(dev_it, [row[key] for row in development], marker="s",
                            linestyle="--", markersize=3, label=label, color=color)
    axes[1, 0].set(xlabel="Updates; completed full-attempt denominator below",
                   ylabel="Full-attempt fraction / soft score" if has_whole_body else "Full-attempt fraction",
                   ylim=(-.05, 1.05), title="Development outcomes and recovery diagnostics")
    if len(development) <= 12:
        axes[1, 0].set_xticks(dev_it, [f"{r['iteration']}\nn={r['completed']}" for r in development], fontsize=7)
    axes[1, 0].legend(fontsize=7, loc="best")
    mean = [r["signed_error_mean_ms"] if r["signed_error_mean_ms"] is not None else np.nan for r in development]
    lo = [r["signed_error_p05_ms"] if r["signed_error_p05_ms"] is not None else np.nan for r in development]
    hi = [r["signed_error_p95_ms"] if r["signed_error_p95_ms"] is not None else np.nan for r in development]
    axes[1, 1].axhline(0, color="gray", linewidth=.8)
    axes[1, 1].fill_between(dev_it, lo, hi, color="#1f77b4", alpha=.15, label="signed p05–p95")
    axes[1, 1].plot(dev_it, mean, "o-", color="#1f77b4", label="signed mean")
    axes[1, 1].plot(dev_it, [r["absolute_error_mean_ms"] if r["absolute_error_mean_ms"] is not None else np.nan
                            for r in development], "s-", color="#ff7f0e", label="absolute mean")
    axes[1, 1].set(xlabel="Updates; hit denominator below", ylabel="Actual TTC − requested TTC (ms)",
                   title="First-encounter timing, hits only")
    if len(development) <= 12:
        axes[1, 1].set_xticks(dev_it, [f"{r['iteration']}\nh={r['timing_denominator']}" for r in development], fontsize=7)
    axes[1, 1].legend(fontsize=8)
    axes[1, 2].plot(iterations, [row["mean_action_std"] for row in learning], color="#9467bd")
    axes[1, 2].set(xlabel="Completed PPO updates", ylabel="Mean action standard deviation",
                   title="Exploration and adaptive learning rate")
    lr = axes[1, 2].twinx()
    lr.plot(iterations, [row["learning_rate"] for row in learning], color="#7f7f7f", alpha=.5)
    lr.set(ylabel="Learning rate", yscale="log")
    for axis in axes.flat:
        axis.grid(alpha=.2)
    fig.suptitle(f"Task 10C jab — {run.name}\nFull target credit uses first encounter; recovery and falls remain separate", fontsize=14)
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)
    flat = [{**{key: value for key, value in row.items() if key != "raw_components"},
             **{f"raw_{name}_mean": value for name, value in row["raw_components"].items()}}
            for row in development]
    if flat:
        with (output / "development.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in flat for key in row)))
            writer.writeheader()
            writer.writerows(flat)
    result = {
        "run": str(run), "last_completed_iteration": int(iterations[-1]),
        "cumulative_transitions": int(transitions[-1]), "elapsed_application_minutes": float(elapsed[-1]),
        "learning_rows_read": len(learning), "ignored_incomplete_final_line_bytes": partial_bytes,
        "plotted_reward_components": component_names,
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "development_panel_count": len(development), "development": development,
        "baseline": development[0] if development else None,
        "latest_development": development[-1] if development else None,
        "same_development_recipes": len({row["recipe_sha256"] for row in development}) <= 1,
        "source_snapshots": snapshots,
        "interpretation": [
            "Training component curves use all logged full attempts and snippets; their mixture is explicit and is not a full-cycle success rate.",
            "Development outcomes use completed full attempts; timing errors use hit attempts only. Null timing at zero hits is preserved.",
            "A hit followed by a fall counts in both columns. Timed encounters alone do not demonstrate usable recovery.",
            "Hand/pelvis settling is the limited historical glove/upright diagnostic; it does not establish whole-body guard or foot support.",
            "Whole-body tracking and hold support quality are soft diagnostics, not success fractions or acceptance gates. Native left/right vertical loads and positive-load durations remain separately reported.",
            "Maximum commanded-target excess is measured outside named hard model limits in radians; its episode maxima and actual command_excess reward are distinct columns.",
            "This read-only analysis snapshot may precede the end of an active run; it does not claim a terminal job status.",
        ],
    }
    (output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rolling-window", type=int, default=25)
    args = parser.parse_args()
    if args.rolling_window < 1:
        parser.error("--rolling-window must be positive")
    result = analyze(args.run, args.output, args.rolling_window)
    print(json.dumps({key: result[key] for key in (
        "run", "last_completed_iteration", "cumulative_transitions", "development_panel_count")}, indent=2))


if __name__ == "__main__":
    main()
