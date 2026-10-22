"""Return each hook to its own initial upper-body guard on its recovered stance.

Only recovery arm/head coordinates change. The original launch, outward punch,
strike, root/legs/waist trajectory and world orientation are retained. Appended
samples are explicitly generated return/hold samples, not new captured motion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import BPoly

from .hook_correction import smoothstep
from .references import StoredReference, arrays_from_qpos
from .robot import JOINT_NAMES, ROOT

OUTPUT_ROOT = ROOT / "outputs/task10c/reference_correction/hooks_final/initial_guard_return"
COMPACT_OUTPUT_ROOT = ROOT / "outputs/task10c/reference_correction/hooks_final/right_compact_return"


def correct_guard_return(input_npz, output_npz):
    input_npz, output_npz = Path(input_npz).resolve(), Path(output_npz).resolve()
    output_npz.relative_to(OUTPUT_ROOT.resolve())
    reference = StoredReference.load(input_npz)
    family = reference.metadata["family"]
    if family not in ("left_hook", "right_hook"):
        raise ValueError("Initial upper-guard return is limited to the two hook references")
    old_t = reference.time_s
    dt = float(np.median(np.diff(old_t)))
    if not np.allclose(np.diff(old_t), .02, atol=1e-10):
        raise ValueError("Expected the selected 50Hz reference grid")
    old_end = float(old_t[-1])
    extension_steps, return_steps = 18, 12
    time_s = np.r_[old_t, old_end + np.arange(1, extension_steps + 1) * dt]
    qpos = np.concatenate((reference.arrays["qpos"],
                           np.repeat(reference.arrays["qpos"][-1:], extension_steps, axis=0)))
    names = [name for name in JOINT_NAMES
             if any(part in name for part in ("shoulder_", "elbow_", "wrist_", "head_"))]
    addresses = np.array([7 + JOINT_NAMES.index(name) for name in names])
    initial_guard = reference.arrays["qpos"][0, addresses]
    start = max(old_end - .45, float(reference.metadata["strike_time_s"]) + .20)
    reached = old_end + return_steps * dt
    for i, t in enumerate(time_s):
        weight = smoothstep((t - start) / (reached - start))
        qpos[i, addresses] = (1. - weight) * qpos[i, addresses] + weight * initial_guard
    arrays = arrays_from_qpos(time_s, qpos)
    arrays["source_frames"] = np.r_[reference.arrays["source_frames"],
                                     np.repeat(reference.arrays["source_frames"][-1], extension_steps)]
    details = {
        "revision": "initial_upper_guard_return_v1",
        "input_reference": str(input_npz),
        "input_sha256": hashlib.sha256(input_npz.read_bytes()).hexdigest(),
        "method": "One smooth quintic blend of arm/head joint posture into this family's corrected initial guard, on the existing recovered root/legs/waist",
        "recovery_change_start_s": float(start),
        "terminal_hold_start_s": float(reached),
        "zero_stored_velocity_hold_start_s": float(reached + dt),
        "original_duration_s": old_end,
        "duration_s": float(time_s[-1]),
        "extension_s": float(time_s[-1] - old_end),
        "changed_joint_names": names,
        "initial_guard_time_s": float(old_t[0]),
        "initial_guard_source_frame": float(reference.arrays["source_frames"][0]),
        "upper_body_goal": "Exact initial arm/head posture relative to retained torso/head; no full world-pose reset",
        "preserved_coordinates": "Original pelvis position/quaternion, leg joints and waist at all existing times; their recovered final values are held through the extension",
        "clock": "Launch t=0 and nominal strike unchanged; all original timestamps retained, only recovery duration extended",
        "source_lineage": {
            "original_frame_array_preserved": True,
            "appended_samples": extension_steps,
            "appended_repeated_source_frame": float(reference.arrays["source_frames"][-1]),
            "meaning": "Repeated last source frame denotes inherited lineage, not new recorded time or a captured held pose. The appended return/hold is generated; its upper-body endpoint comes from initial_guard_source_frame, while its stance/torso comes from the recovered input endpoint.",
        },
        "parent_annotation_status": "Inherited guard_correction and hook_final_correction describe the input trajectory; this operation supersedes their terminal upper-body guard claims",
        "discrete_velocity_note": "Central-difference velocities are rebuilt from the complete new qpos path. The first constant-pose sample can retain its preceding-sample derivative; the subsequent stationary tail has zero velocity.",
    }
    metadata = {**reference.metadata, "hook_guard_return": details,
                "end_time_s": float(time_s[-1]),
                "recovery_duration_s": float(time_s[-1] - reference.metadata["strike_time_s"]),
                "terminal_guard_interval_s": [float(reached + dt), float(time_s[-1])],
                "withdrawal_interval_s": [float(reference.metadata["strike_time_s"]), float(reached)],
                "terminal_guard_upper_body_source_frame": float(reference.arrays["source_frames"][0]),
                "annotation_rationale": "Launch and outward strike annotation are unchanged. Each hook recovers to its own corrected initial arm/head posture on its existing recovered stance/torso. Appended source-frame values record lineage only; no captured source time is invented."}
    metadata.pop("terminal_guard_source_frame", None)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    output_npz.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return details


def compact_right_return(input_npz, output_npz, *, folded_waypoint=False):
    """Replace the moving-parent return with one coordinated fixed waypoint.

    This explicit RH-only mode leaves the original initial-guard return recipe
    reproducible. No optimization, inverse kinematics, or candidate search is
    performed: two fixed quintic Hermite spans determine seven arm coordinates.
    """
    input_npz, output_npz = Path(input_npz).resolve(), Path(output_npz).resolve()
    output_npz.relative_to(COMPACT_OUTPUT_ROOT.resolve())
    reference = StoredReference.load(input_npz)
    if reference.metadata["family"] != "right_hook":
        raise ValueError("The compact recovery mode accepts only right_hook")
    times = reference.time_s
    names = [name for name in JOINT_NAMES if name.startswith("right_")
             and any(part in name for part in ("shoulder_", "elbow_", "wrist_"))]
    addresses = np.array([7 + JOINT_NAMES.index(name) for name in names])
    velocity_indices = np.array([6 + JOINT_NAMES.index(name) for name in names])
    start, middle, end = (.90, 1.10, 1.58) if folded_waypoint else (.74, 1.02, 1.58)
    if folded_waypoint and reference.metadata.get("right_hook_compact_recovery", {}).get("revision") != "right_compact_fixed_waypoint_v1":
        raise ValueError("The folded-waypoint refinement requires the preserved first compact return")
    index = int(np.argmin(np.abs(times - start)))
    if abs(times[index] - start) > 1e-8 or end > times[-1]:
        raise ValueError("Selected RH time grid does not contain the intended recovery interval")
    start = float(times[index])
    anchor = reference.arrays["qpos"][index, addresses].copy()
    guard = reference.arrays["qpos"][-1, addresses].copy()
    incoming_velocity = reference.arrays["qvel"][index, velocity_indices].copy()
    waypoint_velocity = (guard - anchor) / (end - start)
    waypoint = anchor + (middle - start) * waypoint_velocity
    # As roll crosses -90 degrees, move pitch through zero rather than keeping
    # it positive and sending the upper arm backward. The optional final
    # refinement preserves the known-clear entry before folding the forearm.
    waypoint[:4] = [-.30, -np.pi / 2., -.60, -.60] if folded_waypoint else [0., -np.pi / 2., -.60, 0.]
    polynomial = BPoly.from_derivatives(
        [start, middle, end],
        [[anchor, incoming_velocity, np.zeros(7)],
         [waypoint, waypoint_velocity, np.zeros(7)],
         [guard, np.zeros(7), np.zeros(7)]],
    )
    qpos = reference.arrays["qpos"].copy()
    for i, time in enumerate(times):
        if start < time < end:
            qpos[i, addresses] = polynomial(float(time))
        elif time >= end:
            qpos[i, addresses] = guard
    arrays = arrays_from_qpos(times, qpos)
    arrays["source_frames"] = reference.arrays["source_frames"].copy()
    details = {
        "revision": "right_compact_folded_waypoint_v2" if folded_waypoint else "right_compact_fixed_waypoint_v1",
        "input_reference": str(input_npz),
        "input_sha256": hashlib.sha256(input_npz.read_bytes()).hexdigest(),
        "edited_interval_s": [start, end if folded_waypoint else float(reference.metadata["hook_guard_return"]["terminal_hold_start_s"])],
        "transition_interval_s": [start, end],
        "active_arm_guard_reached_s": end,
        "whole_upper_body_guard_reached_s": float(reference.metadata["hook_guard_return"]["terminal_hold_start_s"]),
        "joint_names": names,
        "construction": "Two fixed-endpoint quintic Hermite spans through one explicitly coordinated shoulder/elbow waypoint; no moving baseline blend, optimizer, planner, or search",
        "waypoint_time_s": middle,
        "anchor_joint_position_rad": anchor.tolist(),
        "waypoint_joint_position_rad": waypoint.tolist(),
        "guard_joint_position_rad": guard.tolist(),
        "anchor_velocity_rad_s": incoming_velocity.tolist(),
        "waypoint_velocity_rad_s": waypoint_velocity.tolist(),
        "joins": "Anchor matches input position and its stored central-difference velocity; spans share waypoint position/velocity/zero acceleration; final guard has zero velocity/acceleration",
        "preserved": "All timestamps, source-frame lineage, launch/strike, initial/outward motion, pelvis/root, legs, torso/waist, head, left arm, and final own-initial guard/hold",
        "source_lineage": "The edited recovery is a procedural joint transition. Retained source_frames identify parent lineage, not newly captured waypoint poses.",
        "purpose": "Move shoulder pitch through zero with the roll unwind and coordinate elbow folding, replacing the inherited backward elbow detour rather than only slowing it",
        "local_repair": ("Preserve the first compact candidate through .90s, including its narrow but clear .84-.88s entry; then use one waypoint with shoulder pitch -.30rad and elbow -.60rad to fold the forearm during withdrawal. No additional waypoint or search." if folded_waypoint else "Entry is at .74s after shoulder roll stops approaching its limit; the single middle elbow target is 0rad to clear the unchanged inactive glove. No additional waypoint or search."),
    }
    if folded_waypoint:
        entry = reference.metadata["right_hook_compact_recovery"]
        details["edited_interval_s"] = entry["edited_interval_s"]
        details["refinement_interval_s"] = [start, end]
        details["retained_entry_interval_s"] = [entry["transition_interval_s"][0], start]
        details["retained_entry_construction"] = entry
        details["construction"] = "Retain the first compact Hermite entry from .74 to .90s, then use two fixed quintic Hermite spans through one folded waypoint; no optimizer, planner, or search"
    metadata = {**reference.metadata, "right_hook_compact_recovery": details,
                "withdrawal_interval_s": [float(reference.metadata["strike_time_s"]), end],
                "annotation_rationale": "Launch, strike, source lineage and duration unchanged. RH active arm uses a fixed coordinated return and reaches its own initial guard at1.58s. The untouched left arm finishes its existing return at1.74s, so the existing full two-hand stationary guard interval remains correct. Inherited hook_guard_return records the parent construction, superseded only for the RH arm by right_hook_compact_recovery."}
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    output_npz.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact-right", action="store_true",
                        help="RH-only fixed-waypoint recovery replacement; original return mode remains the default.")
    parser.add_argument("--folded-waypoint", action="store_true",
                        help="Refine the preserved first compact return after its clear entry; requires --compact-right.")
    args = parser.parse_args()
    if args.folded_waypoint and not args.compact_right:
        parser.error("--folded-waypoint requires --compact-right")
    details = (compact_right_return(args.input, args.output, folded_waypoint=args.folded_waypoint)
               if args.compact_right else correct_guard_return(args.input, args.output))
    print(json.dumps(details, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
