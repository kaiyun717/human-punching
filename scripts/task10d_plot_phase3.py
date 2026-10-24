#!/usr/bin/env python3
"""Export Phase3 geometric-search figures; no renderer or coverage execution.

Run through scripts/task10d_env.sh with the existing .venv-task10c interpreter.
The source identities and numerical summaries are retained beside the figures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
LABELS = {
    "jab_nominal": "Jab · nominal", "cross_nominal": "Cross · nominal",
    "left_hook_nominal": "Left hook · nominal", "right_hook_nominal": "Right hook · nominal",
    "jab_004_01107": "Stepping jab", "cross_012_01441": "Compact cross",
    "left_hook_013_03243": "Tight left hook", "right_hook_014_02074": "Tight right hook",
}
BLUE, GRAY = "#14669b", "#666b73"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_figure(fig, output: Path, name: str) -> list[str]:
    paths = []
    for extension in ("png", "pdf"):
        path = output / f"{name}.{extension}"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite figure: {path}")
        fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
        paths.append(path.name)
    plt.close(fig)
    return paths


def availability_figure(summary: dict, output: Path) -> list[str]:
    primitives = summary["primitive_ids"]
    pitches = np.array([row["pitch_up_deg"] for row in summary["pitches"]])
    count = summary["state_count"]
    candidate = np.array([[row["per_primitive_available"][name] for row in summary["pitches"]]
                          for name in primitives]) * 100 / count
    baseline = np.array([[row["per_primitive_baseline"][name] for row in summary["pitches"]]
                         for name in primitives]) * 100 / count
    fig = plt.figure(figsize=(13.0, 9.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[.8, 2], hspace=.40, wspace=.26,
                          left=.16, right=.96, top=.87, bottom=.15)
    ax = fig.add_subplot(grid[0, :])
    for key, label, color, linestyle in (
            ("states_with_attack", "Bounded placement search", BLUE, "-"),
            ("baseline_states_with_attack", "Accepted placement unchanged", GRAY, "--")):
        values = np.array([row[key] for row in summary["pitches"]]) * 100 / count
        ax.plot(pitches, values, marker="o", markersize=4, lw=2, color=color,
                linestyle=linestyle, label=label)
    ax.set(xlim=(pitches[0] - 1, pitches[-1] + 1), ylim=(-3, 103),
           xticks=pitches, yticks=[0, 25, 50, 75, 100], ylabel="States with ≥1 attack (%)")
    ax.grid(alpha=.18)
    ax.legend(frameon=False, loc="lower right", ncol=2)
    for column, (matrix, title) in enumerate(((candidate, "Bounded placement search"),
                                             (baseline, "Accepted placement unchanged"))):
        heat = fig.add_subplot(grid[1, column])
        heat.imshow(matrix, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        heat.set(xticks=range(len(pitches)), xticklabels=[f"{x:+g}" for x in pitches],
                 yticks=range(len(primitives)), xlabel="Mounting pitch (degrees; positive up)", title=title)
        heat.set_yticklabels([LABELS[name] for name in primitives] if column == 0 else [""] * len(primitives))
        heat.tick_params(axis="both", length=0)
        for row in range(len(primitives)):
            for col in range(len(pitches)):
                value = matrix[row, col]
                text = f"{value:.0f}" if value in (0, 100) else f"{value:.1f}"
                heat.text(col, row, text, ha="center", va="center", fontsize=8,
                          color="white" if value > 58 else "#172c3b")
        heat.set_xticks(np.arange(-.5, len(pitches), 1), minor=True)
        heat.set_yticks(np.arange(-.5, len(primitives), 1), minor=True)
        heat.grid(which="minor", color="white", linewidth=.6)
        heat.tick_params(which="minor", length=0)
    fig.suptitle("Initial camera geometry and outward reach", x=.55, y=.965, fontsize=18, fontweight="semibold")
    fig.text(.55, .925, f"{count:,} TRAIN design states · all eight accepted motions · cell values are percent of states",
             ha="center", fontsize=11, color="#444444")
    fig.text(.16, .067,
        "Joint geometric availability requires the same placement to satisfy outward reach (≤0.14 m), both initial\n"
        "glove clearances (≥0.16 m), and both glove-center frustum tests. Search also applies the declared frontal bounds.",
        fontsize=9, color="#444444", linespacing=1.45)
    fig.text(.16, .018,
        "These figures do not establish clean observation or initial body-intersection validity.\n"
        "The posture-covering design subset is not a frequency-representative sample of the full bank.",
        fontsize=9, color="#444444", linespacing=1.45)
    return save_figure(fig, output, "geometric_availability_vs_pitch")


def _quantiles(values: np.ndarray) -> dict:
    return dict(zip(("min", "p05", "p25", "median", "p75", "p95", "max"),
                    np.quantile(values, [0, .05, .25, .5, .75, .95, 1]).tolist()))


def margin_figure(summary: dict, records: dict, output: Path, pitches: list[float]) -> tuple[list, dict]:
    available_pitches = np.asarray([row["pitch_up_deg"] for row in summary["pitches"]])
    if len(pitches) != 6:
        raise ValueError("The margin figure accepts exactly six comparison pitches")
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 8.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.97, top=.81, bottom=.20, hspace=.40, wspace=.18)
    statistics = {}
    all_data = []
    for ax, pitch in zip(axes.flat, pitches, strict=True):
        positions = np.flatnonzero(available_pitches == pitch)
        if len(positions) != 1:
            raise ValueError(f"Requested pitch is absent: {pitch}")
        p = positions[0]
        optimized, baseline = [], []
        per_primitive = {}
        for name in summary["primitive_ids"]:
            record = records[name]
            keep = record["top_available"][p, :, 0]
            candidate_margin = record["top_center_margin"][p, :, 0].min(-1)[keep]
            original_margin = record["baseline_center_margin"][p].min(-1)[keep]
            if not np.isfinite(candidate_margin).all() or not np.isfinite(original_margin).all():
                raise ValueError("Nonfinite camera margins")
            optimized.append(candidate_margin); baseline.append(original_margin)
            per_primitive[name] = int(keep.sum())
        optimized, baseline = np.concatenate(optimized), np.concatenate(baseline)
        if not len(optimized):
            ax.text(.5, .5, "No admitted candidate", transform=ax.transAxes, ha="center")
            continue
        cdf = np.arange(1, len(optimized) + 1) / len(optimized)
        ax.plot(np.sort(optimized), cdf, color=BLUE, lw=1.8)
        ax.plot(np.sort(baseline), cdf, color=GRAY, ls="--", lw=1.8)
        ax.axvline(0, color="#a24b3c", lw=1, alpha=.65)
        ax.grid(alpha=.15)
        ax.set_title(f"{pitch:+g}° · {len(optimized):,} matched pairs", fontsize=11)
        ax.set(ylim=(0, 1.02), yticks=[0, .25, .5, .75, 1])
        statistics[str(pitch)] = dict(pair_count=len(optimized),
            per_primitive_pair_count=per_primitive, candidate=_quantiles(optimized),
            literal_baseline_same_pairs=_quantiles(baseline),
            baseline_margin_nonnegative_pairs=int((baseline >= 0).sum()))
        all_data.extend((optimized, baseline))
    left = float(np.floor(min(values.min() for values in all_data) * 5) / 5)
    right = float(np.ceil(max(values.max() for values in all_data) * 5) / 5)
    for ax in axes.flat:
        ax.set_xlim(left - .02, right + .02)
    for ax in axes[:, 0]:
        ax.set_ylabel("Empirical cumulative fraction")
    for ax in axes[-1]:
        ax.set_xlabel("Worst-hand normalized image-boundary margin")
    fig.suptitle("Initial glove-center margins at matched reset states", y=.965, fontsize=18, fontweight="semibold")
    fig.text(.53, .915, "One top-ranked placement per available state / primitive; two hands reduced to their worse margin",
             ha="center", fontsize=11, color="#444444")
    fig.legend(handles=[Line2D([0], [0], color=BLUE, lw=2, label="Selected bounded placement"),
                        Line2D([0], [0], color=GRAY, ls="--", lw=2, label="Accepted placement at the same state / primitive")],
               loc="upper center", bbox_to_anchor=(.53, .89), frameon=False, ncol=2)
    fig.text(.09, .105,
        "Margin = min over both hands and image axes of 1 − |coordinate| / (forward depth × tan(half FOV)).\n"
        "Zero is the center-admission boundary; negative baseline values are outside the angular frustum.",
        fontsize=9, color="#444444", linespacing=1.45)
    fig.text(.09, .035,
        "Each panel conditions on candidate geometric availability and compares exactly the same state / primitive pairs.\n"
        "Counts and conditioning vary with pitch. These are center-projection margins, not clean-pixel or full-sphere guarantees.",
        fontsize=9, color="#444444", linespacing=1.45)
    return save_figure(fig, output, "worst_hand_margin_distributions"), statistics


def snapshot_review(summary: dict) -> dict:
    path = Path(summary["geometry_path"])
    if sha256(path) != summary["geometry_sha256"]:
        raise ValueError("Search snapshot geometry identity changed")
    subset_path = Path(summary["subset_path"])
    if sha256(subset_path) != summary["subset_sha256"]:
        raise ValueError("Search training-subset identity changed")
    subset = json.loads(subset_path.read_text())
    with np.load(path, allow_pickle=False) as z:
        ids = z["snapshot_id"]
        selected = np.searchsorted(ids, subset["snapshot_ids"])
        train = np.flatnonzero(z["split_id"] == 0)
        features = z["features"]
        nominal = int(np.searchsorted(ids, 53874))
        ranges = {}
        for index, name in enumerate(subset["feature_names"]):
            values = features[:, index]
            ranges[name] = dict(train_min=float(values[train].min()), train_max=float(values[train].max()),
                selected_min=float(values[selected].min()), selected_max=float(values[selected].max()),
                min_snapshot_id=int(ids[selected[np.argmin(values[selected])]]),
                max_snapshot_id=int(ids[selected[np.argmax(values[selected])]]),
                nominal_value=float(values[nominal]))
        # Signed waist heading remains readable; the design features use sin/cos.
        yaw = np.arctan2(features[:, 5], features[:, 6])
        velocities = z["sim/data/qvel"]
        measured = z["actual_velocity"]
        return dict(feature_ranges=ranges,
            training_extrema_retained=bool(np.array_equal(features[selected].min(0), features[train].min(0))
                                          and np.array_equal(features[selected].max(0), features[train].max(0))),
            waist_heading_deg=dict(train_range=np.rad2deg([yaw[train].min(), yaw[train].max()]).tolist(),
                selected_range=np.rad2deg([yaw[selected].min(), yaw[selected].max()]).tolist()),
            nominal=dict(snapshot_id=53874, head_world=z["head_world"][nominal].tolist(),
                camera_position_world=z["camera_position_world"][nominal].tolist(),
                measured_body_planar_and_gyro_z_velocity=measured[nominal].tolist(),
                source_free_root_qvel=velocities[nominal, :6].tolist(),
                generalized_velocity_norm=float(np.linalg.norm(velocities[nominal])),
                maximum_absolute_joint_velocity=float(np.max(np.abs(velocities[nominal, 6:])))),
            selected_velocity=dict(generalized_velocity_norm_quantiles=_quantiles(np.linalg.norm(velocities[selected], axis=1)),
                zero_generalized_velocity_rows=int(np.count_nonzero(np.linalg.norm(velocities[selected], axis=1) == 0))),
            movement_scope="These are saved reset velocities and static FK posture ranges, not integrated movement or dynamic replay results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path, default=PROJECT / "outputs/task10d/phase3/search01")
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/task10d/phase3/figures01")
    parser.add_argument("--margin-pitches", type=float, nargs=6, default=[-15, 15, 20, 25, 30, 35])
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((PROJECT / "outputs/task10d/phase3").resolve()):
        raise ValueError("Figure outputs must remain within human-punching Phase3")
    output.mkdir(parents=True, exist_ok=False)
    summary_path = args.search.resolve() / "summary.json"
    summary = json.loads(summary_path.read_text())
    records, identities = {}, {}
    for primitive, name in zip(summary["primitive_ids"], summary["candidate_files"], strict=True):
        path = summary_path.parent / name
        with np.load(path, allow_pickle=False) as source:
            records[primitive] = {key: source[key].copy() for key in
                ("snapshot_id", "available", "pitches_up_deg", "top_available", "top_center_margin", "baseline_center_margin")}
        identities[name] = sha256(path)
        expected = [row["per_primitive_available"][primitive] for row in summary["pitches"]]
        np.testing.assert_array_equal(records[primitive]["available"].sum(-1), expected)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42})
    figures = availability_figure(summary, output)
    margin_files, margins = margin_figure(summary, records, output, args.margin_pitches)
    figures.extend(margin_files)
    review = snapshot_review(summary)
    write_json(output / "snapshot_feature_review.json", review)
    manifest = dict(schema="task10d_phase3_geometric_figure_manifest_v1", search=str(summary_path.parent),
        summary_sha256=sha256(summary_path), candidate_sha256=identities,
        script_sha256=sha256(Path(__file__)), figures={name: sha256(output / name) for name in figures},
        margin_statistics=margins, margin_pitches_up_deg=args.margin_pitches,
        result_scope="Geometric outward-reach plus initial-center-frustum search only; no claim of clean observation or verified initial body intersections",
        population="1000 posture-covering TRAIN design states; no iid/generalization claim",
        command="source scripts/task10d_env.sh && .venv-task10c/bin/python scripts/task10d_plot_phase3.py"
                f" --search {args.search} --output {args.output}")
    write_json(output / "manifest.json", manifest)
    print(json.dumps(dict(figures=figures, training_extrema_retained=review["training_extrema_retained"],
                         nominal=review["nominal"]), indent=2))


if __name__ == "__main__":
    main()
