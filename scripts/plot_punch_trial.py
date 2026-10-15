#!/usr/bin/env python3
"""Plot independent FK and recorded physical evidence for one punch trial."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from human_punching.jab_evaluation import (
    _kinematics,
    _load_arrays,
    _reference_at,
    _reference_metadata,
    _structure,
    _validate_trajectory,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model))
    structure = _structure(model)
    arrays, reference = _load_arrays(args.rollout), _load_arrays(args.reference)
    metadata = _reference_metadata(reference, args.reference)
    times, poses, velocities = _validate_trajectory(model, arrays, structure)
    speed = float(np.asarray(arrays.get("reference_speed", 1)).item())
    ref_q, ref_v = _reference_at(model, reference, times, structure, speed)
    actual = _kinematics(model, poses, velocities, structure)
    desired = _kinematics(model, ref_q, ref_v, structure)
    head = np.asarray(reference.get("head_center", reference.get("head_center_w")))
    active = {"left": 0, "right": 1}[str(metadata["active_hand"]).lower()]
    distance = np.linalg.norm(actual["gloves"][:, active] - head, axis=1)
    target_distance = np.linalg.norm(desired["gloves"][:, active] - head, axis=1)
    error = np.linalg.norm(actual["gloves"] - desired["gloves"], axis=-1)
    drift = np.linalg.norm(actual["soles"] - actual["soles"][0], axis=-1)
    limits = model.jnt_range[structure["joints"]]
    angles = poses[:, structure["qadr"]]
    violation = np.maximum(np.maximum(limits[:, 0] - angles, angles - limits[:, 1]), 0)
    elapsed = times - times[0]
    window = metadata.get("attack_window_reference_s")
    family = str(metadata.get("punch_family", metadata.get("family", "punch"))).replace("_", " ")
    physics_dt = float(np.asarray(arrays.get("physics_timestep_s", model.opt.timestep)).item())
    action_dt = float(np.asarray(arrays.get("policy_timestep_s", .020)).item())
    if "nonuniform" in str(metadata.get("timing_kind", "")).lower():
        speed_label = f"nonuniformly retimed reference · {speed:g}× playback"
    else:
        source_speed = metadata.get("source_speed")
        speed_label = f"{float(source_speed) * speed:g}× source speed" if source_speed is not None else f"{speed:g}× reference playback"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, layout="constrained")
    for ax in axes.flat:
        ax.grid(alpha=.2)
        if window is not None:
            ax.axvspan(window[0] / speed, window[1] / speed, color="#dbeafe", alpha=.6)
    ax = axes[0, 0]
    ax.plot(elapsed, target_distance, "--", color="#64748b", label="Reference")
    ax.plot(elapsed, distance, color="#2563eb", label="Recorded active glove")
    ax.axhline(.14, color="#dc2626", linewidth=1, label="Contact: 0.14 m")
    ax.axhline(.16, color="#16a34a", linewidth=1, linestyle=":", label="Safe clearance: 0.16 m")
    ax.set(ylabel="Glove–head center distance (m)", title="Attack and recovery timing")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for index, label in enumerate(("Left glove", "Right glove")):
        ax.plot(elapsed, error[:, index] * 100, label=label)
    ax.axhline(2, color="#64748b", linestyle=":", label="2 cm RMS target (scale reference)")
    ax.set(ylabel="Instantaneous position error (cm)", title="World-space glove tracking")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    for index, label in enumerate(("Left sole", "Right sole")):
        ax.plot(elapsed, drift[:, index] * 100, label=label)
    ax.axhline(3, color="#dc2626", linestyle=":", label="3 cm gate")
    ax.set(ylabel="Sole drift from initial position (cm)", xlabel="Time after guard reset (s)", title="Foot stability")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.plot(elapsed, violation.max(axis=1), color="#9333ea", label="Maximum across joints")
    ax.axhline(.02, color="#dc2626", linestyle=":", label="0.02 rad gate")
    ax.set(ylabel="Joint-limit excursion (rad)", xlabel="Time after guard reset (s)", title="Soft-limit audit")
    ax.legend(fontsize=8)
    fig.suptitle(f"{family.title()}: recorded physics against reference\n"
                 f"{physics_dt * 1000:g} ms physics · {action_dt * 1000:g} ms actions · {speed_label} · shaded attack window", fontsize=14)
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(args.output / f"trial-evidence.{extension}", dpi=150)
    plt.close(fig)
    (args.output / "plot-provenance.json").write_text(json.dumps({
        "model": str(args.model), "reference": str(args.reference), "rollout": str(args.rollout),
        "method": "Forward kinematics of every recorded physics state; reference interpolated on the same clock. No dynamics replay, trajectory editing, or smoothing.",
        "attack_window_reviewed": bool(metadata.get("attack_window_reviewed", False)),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
