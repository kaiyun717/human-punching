#!/usr/bin/env python3
"""Review derived attack windows against dense FK; this is not a physics rollout."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from human_punching.contact import first_contact_time
from human_punching.physical import interpolate_pose


def contact_intervals(times, distance, threshold=.14):
    """Threshold episodes, with linearly interpolated radial-distance crossings."""
    inside = distance <= threshold
    starts = np.flatnonzero(inside & ~np.r_[False, inside[:-1]])
    ends = np.flatnonzero(inside & ~np.r_[inside[1:], False])

    def crossing(i, j):
        return float(times[i] + (threshold-distance[i]) /
                     (distance[j]-distance[i]) * (times[j]-times[i]))

    return [[crossing(start-1, start) if start else float(times[start]),
             crossing(end, end+1) if end+1 < len(times) else float(times[end])]
            for start, end in zip(starts, ends, strict=True)]


def review(model_path, reference_path, output):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    with np.load(reference_path, allow_pickle=False) as archive:
        ref = dict(archive)
    original_time = ref["time_s"]
    times = np.r_[np.arange(0., original_time[-1], .005), original_time[-1]]
    hands = [model.site(f"{side}Hand_center").id for side in ("Left", "Right")]
    gloves = []
    for t in times:
        data.qpos[:] = interpolate_pose(model, original_time, ref["qpos"], t)[0]
        mujoco.mj_forward(model, data)
        gloves.append(data.site_xpos[hands].copy())
    gloves = np.array(gloves)
    head = ref["head_center"]
    distance = np.linalg.norm(gloves-head, axis=-1)
    radial = np.gradient(distance, times, axis=0)
    velocity = np.gradient(gloves, times, axis=0)
    active = int(ref["active_hand_index"])
    inactive = 1-active
    launch = float(ref["launch_time"])
    recovery = float(ref["return_start"])
    guard = float(ref["guard_hold_start"])
    window = ref["attack_window_reference_s"].astype(float)
    episodes = contact_intervals(times, distance[:, active])
    first_contact = float(first_contact_time(times, gloves[:, [active, active]], head))
    closest = int(np.argmin(distance[:, active]))
    at_first = int(np.argmin(np.abs(times-first_contact)))
    source_times = np.interp(times, original_time, ref["source_time_s"])
    inside_window = (times >= window[0]) & (times <= window[1])
    after_window = times > window[1]
    checks = {
        "window_begins_at_playback_launch": bool(np.isclose(window[0], launch)),
        "window_ends_before_inserted_recovery": bool(window[1] < recovery),
        "active_contact_is_single_episode": len(episodes) == 1,
        "all_active_contact_episodes_inside_window": bool(episodes and all(
            a >= window[0] and b <= window[1] for a, b in episodes)),
        "first_contact_is_inbound": bool(radial[at_first, active] < 0),
        "closest_approach_is_inside_window": bool(inside_window[closest]),
        "window_end_is_after_closest_approach": bool(window[1] > times[closest]),
        "inactive_hand_never_contacts_head": bool(distance[:, inactive].min() > .14),
        "active_hand_stays_safe_after_window": bool(distance[after_window, active].min() >= .16),
        "active_hand_stays_safe_in_inserted_recovery": bool(distance[times >= recovery, active].min() >= .16),
    }
    report = {
        "schema": "derived_reference_attack_window_review_v1",
        "reference": str(Path(reference_path).resolve()),
        "reference_sha256": hashlib.sha256(Path(reference_path).read_bytes()).hexdigest(),
        "model": str(Path(model_path).resolve()),
        "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        "punch_family": str(ref["punch_family"].item()),
        "active_hand": str(ref["active_hand"].item()),
        "review_scope": "Dense 5ms FK sampling of frozen configuration trajectory; active/inactive distance, approach velocity, derived attack window and inserted recovery. Not source-capture annotation review, dynamic execution, or timing tolerance acceptance.",
        "archive_attack_window_reviewed": bool(ref["attack_window_reviewed"]),
        "derived_window_review_pass": all(checks.values()),
        "checks": checks,
        "attack_window_reference_s": window.tolist(),
        "launch_time_s": launch,
        "inserted_recovery_start_s": recovery,
        "quiet_guard_start_s": guard,
        "active_contact_episodes_s": episodes,
        "active_first_contact_s_quadratic_segment": first_contact,
        "archived_first_contact_s": float(ref["active_first_contact_time"]),
        "dense_minus_archived_first_contact_s": first_contact-float(ref["active_first_contact_time"]),
        "active_first_contact_source_time_s": float(np.interp(first_contact, times, source_times)),
        "radial_speed_at_first_contact_m_s": float(radial[at_first, active]),
        "active_glove_speed_at_first_contact_m_s": float(np.linalg.norm(velocity[at_first, active])),
        "closest_approach_s": float(times[closest]),
        "closest_approach_distance_m": float(distance[closest, active]),
        "minimum_inactive_distance_m": float(distance[:, inactive].min()),
        "minimum_active_distance_after_window_m": float(distance[after_window, active].min()),
        "minimum_active_distance_in_inserted_recovery_m": float(distance[times >= recovery, active].min()),
        "window_end_radial_speed_m_s": float(np.interp(window[1], times, radial[:, active])),
        "method_notes": ["5ms denotes FK audit sample spacing, not simulated physics.",
                         "Episode boundaries interpolate radial distance between adjacent dense FK samples.",
                         "First-contact field uses the common piecewise-linear glove-center quadratic crossing method.",
                         "Frozen archive metadata and bytes are not altered by this review."],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "review.json").write_text(json.dumps(report, indent=2)+"\n")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
    axes[0].plot(times, distance[:, active], label="Active glove / head")
    axes[0].plot(times, distance[:, inactive], label="Inactive glove / head", alpha=.6)
    axes[0].axhline(.14, color="red", linestyle="--", label="Contact 0.14 m")
    axes[0].axhline(.16, color="darkorange", linestyle=":", label="Safety 0.16 m")
    axes[0].set_ylabel("Center distance (m)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].plot(times, radial[:, active], label="Active radial speed")
    axes[1].axhline(0, color="black", linewidth=.6)
    axes[1].set_ylabel("Radial speed (m/s)\nnegative = approaching")
    axes[1].set_xlabel("Retimed archive time since reset (s)")
    for ax in axes:
        ax.axvspan(window[0], window[1], alpha=.10, color="tab:blue", label="Derived attack window")
        ax.axvspan(recovery, guard, alpha=.12, color="tab:gray", label="Inserted recovery")
        ax.axvline(first_contact, color="red", linewidth=.8)
        ax.grid(alpha=.2)
    fig.suptitle(f"{report['punch_family'].replace('_', ' ').title()}: kinematic attack-window review")
    fig.savefig(output / "distance-and-radial-speed.png", dpi=160)
    plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", nargs="+", default=["left_hook", "right_hook"])
    args = parser.parse_args()
    reports = [review(args.model, args.bank/family/f"{family}.npz", args.output/family)
               for family in args.families]
    (args.output / "summary.json").write_text(json.dumps(reports, indent=2)+"\n")
    (args.output / "review_reference_windows_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
