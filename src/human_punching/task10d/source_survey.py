"""Focused source-event survey for the sixteen single-family DuoBox recordings.

Source reach peaks nominate events; they are not automatic punch labels or H2
quality gates. Only the explicit event manifest is eligible for later conversion.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from human_punching.motion import FPS, MotionClip, PARENTS, compose_global_rotations, forward_kinematics
from human_punching.task10c.robot import ROOT

FAMILY = {"jab": "jab", "cross": "cross", "leadhook": "left_hook", "rearhook": "right_hook"}
EXPECTED = {"jab": 7, "cross": 5, "left_hook": 2, "right_hook": 2}
REVIEW_SELECTION = {
    "jab": ("jab_000_00430", "jab_004_01107"),
    "cross": ("cross_003_00292", "cross_012_01441"),
    "left_hook": ("left_hook_006_00464", "left_hook_013_03243"),
    "right_hook": ("right_hook_007_00523", "right_hook_014_02074"),
}
DIVERSITY = {
    "jab_004_01107": "A recorded larger step and lowered terminal pelvis, faster extension, and active-hand recoil with ongoing stance travel; differs from the nearly stationary source of the nominal jab.",
    "cross_012_01441": "A more compact initial right-hand guard, small net pelvis travel, and rising/circular withdrawal from a counter-response drill; same performer and punch family, not a new population style.",
    "left_hook_013_03243": "A tighter bent-arm left hook: 0.477 m maximum source wrist-to-shoulder reach versus 0.555 m in the selected nominal source; different approach curve and bent-arm recovery.",
    "right_hook_014_02074": "A tighter right-hook arm reach (0.500 m versus 0.532 m nominal source) and earlier elbow folding in recorded recovery, with modest pelvis displacement.",
}
# Explicit review edits retain complete recorded stepping/withdrawal and end at
# a quiet guard, rather than cutting solely at the first reach minimum. These
# remain original frame bounds, not synthesized poses or frozen exact footprints.
REVIEW_BOUNDS = {"jab_004_01107": (1023, 1235), "cross_012_01441": (1352, 1535),
                 "left_hook_013_03243": (3161, 3342), "right_hook_014_02074": (2005, 2174)}


def read_recording(path: Path, expected_sha256: str | None = None) -> tuple[dict, str]:
    """Load same hashed bytes from the locally supplied pickle-backed dataset."""
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Source identity changed: {path}")
    data = np.load(io.BytesIO(payload), allow_pickle=True).item()
    for key in ("ma_position", "ma_rotation"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"Nonfinite source {key}: {path}")
    return data, digest


def load_event(event: dict, dataset_root: Path) -> MotionClip:
    """Build a checked MotionClip from explicit inclusive original frame bounds."""
    path = (dataset_root / "fbx_export_joints" / event["source_recording"]).resolve()
    data, digest = read_recording(path, event["source_sha256"])
    first, last = event["source_frame_range_inclusive"]
    positions = np.asarray(data["ma_position"][first:last + 1], dtype=float).copy()
    local = np.asarray(data["ma_rotation"][first:last + 1], dtype=float).copy()
    global_rotations = compose_global_rotations(local)
    offsets = np.zeros_like(positions)
    for j, parent in enumerate(PARENTS[1:], 1):
        offsets[:, j] = np.einsum("tba,tb->ta", global_rotations[:, parent], positions[:, j] - positions[:, parent])
    rest = np.median(offsets, axis=0)
    error = np.linalg.norm(forward_kinematics(positions[:, 0], global_rotations, rest) - positions, axis=-1)
    if float(error.max()) > 1e-4:
        raise ValueError("Source skeleton violates inherited rigid-offset FK convention")
    hand = event["anatomical_active_hand"]
    metadata = {"schema": "human_punching_duobox_crop_v1", "punch": event["family"],
                "candidate_id": event["event_id"], "actor": "ma", "source_path": str(path),
                "source_sha256": digest, "source_rate_hz": FPS,
                "source_frame_range_inclusive": [first, last],
                "selected_frame_range_inclusive": [first, last],
                "outward_completion_frame": event["extension_frame"],
                "anatomical_hand": hand, "active_hand_joint_index": 8 if hand == "left" else 12,
                "original_named_drill": path.name.split("_")[2],
                "source_time_convention": "120 Hz; absolute source frame / 120; crop time zero is first included frame",
                "label_status": "source-drill family with explicit anatomical hand; descriptor-assisted event nomination",
                "fk_max_error_m": float(error.max()), "retargeting": "none; source skeleton only"}
    frames = np.arange(first, last + 1)
    return MotionClip(event["family"], positions, local, global_rotations, rest,
                      np.arange(len(frames)) / FPS, frames, metadata)


def event_row(path: Path, data: dict, digest: str, family: str, peak: int, lo: int, hi: int,
              explicit_bounds: tuple[int, int] | None = None) -> dict:
    p = np.asarray(data["ma_position"], dtype=float)
    hand = 8 if family in ("jab", "left_hook") else 12
    shoulder = hand - 2
    reach = np.linalg.norm(p[:, hand] - p[:, shoulder], axis=1)
    # Each crop includes recorded guards around one extension. Restrict guard
    # search to this event's half-intervals and at most 0.8 s before / 1 s after.
    lo = max(lo, peak - 96)
    hi = min(hi, peak + 120)
    start_search = np.arange(lo, max(lo + 1, peak - 18))
    end_search = np.arange(min(hi - 1, peak + 18), hi + 1)
    first = int(start_search[np.argmin(gaussian_filter1d(reach, 2)[start_search])])
    last = int(end_search[np.argmin(gaussian_filter1d(reach, 2)[end_search])])
    if explicit_bounds is not None:
        first, last = explicit_bounds
    relative = p[:, hand] - p[:, 0]
    v = np.gradient(gaussian_filter1d(relative, 1.5, axis=0), 1 / FPS, axis=0)
    initial_forward = data["ma_rotation"][first, 0] @ np.array([0., 0., 1.])
    yaw = np.arctan2(initial_forward[1], initial_forward[0])
    c, s = np.cos(yaw), np.sin(yaw)
    scene = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])
    path_local = (relative[first:peak+1] - relative[first]) @ scene.T
    chord = np.linalg.norm(path_local[-1])
    travel = np.linalg.norm(np.diff(path_local, axis=0), axis=1).sum()
    rotations = compose_global_rotations(data["ma_rotation"][first:last+1])
    torso_forward = rotations[:, 2] @ np.array([0., 0., 1.])
    torso_yaw = np.unwrap(np.arctan2(torso_forward[:, 1], torso_forward[:, 0]))
    stance_delta = (p[first, 15] - p[first, 19]) @ initial_forward
    bilateral_reach = np.stack([np.linalg.norm(p[first:last+1, h]-p[first:last+1, h-2], axis=1)
                                for h in (8, 12)], axis=1)
    bilateral_excursion = np.ptp(bilateral_reach, axis=0)
    measured_active_hand = "left" if bilateral_excursion[0] > bilateral_excursion[1] else "right"
    complete = bool(peak-first >= 20 and last-peak >= 25 and reach[first] < .40 and reach[last] < .40)
    return {"event_id": f"{family}_{path.name[:3]}_{peak:05d}", "family": family,
            "source_recording": path.name, "source_sha256": digest,
            "source_frame_range_inclusive": [first, last], "extension_frame": int(peak),
            "source_time_convention": "120 Hz; absolute frame / 120; crop clock begins at first included frame",
            "anatomical_active_hand": "left" if hand == 8 else "right",
            "measured_larger_reach_excursion_hand": measured_active_hand,
            "bilateral_wrist_shoulder_reach_excursion_m_left_right": bilateral_excursion.tolist(),
            "hand_evidence_agrees_with_family": measured_active_hand == ("left" if hand == 8 else "right"),
            "stance": "left_foot_forward" if stance_delta > .05 else "right_foot_forward" if stance_delta < -.05 else "nearly_square",
            "initial_left_minus_right_foot_forward_m": float(stance_delta),
            "complete_guard_context_candidate": complete,
            "duration_s": (last-first)/FPS, "preparation_s": (peak-first)/FPS,
            "withdrawal_s": (last-peak)/FPS,
            "source_arm_reach_m": [float(reach[first]), float(reach[peak]), float(reach[last])],
            "active_hand_approach_path_chord_ratio": float(travel/max(chord, 1e-9)),
            "active_hand_peak_speed_pelvis_relative_m_s": float(np.linalg.norm(v[first:last+1], axis=1).max()),
            "active_hand_approach_local_xyz_m": path_local[-1].tolist(),
            "torso_yaw_range_deg": float(np.rad2deg(np.ptp(torso_yaw))),
            "pelvis_net_displacement_m": (p[last, 0]-p[first, 0]).tolist(),
            "active_hand_guard_height_relative_head_m": float(p[first, hand, 2]-p[first, 4, 2]),
            "status": "unselected_repetition" if complete else "exclude_incomplete_guard_context",
            "reason": "Source descriptor candidate; not separately retargeted because repeated drills do not alone add style coverage."
                      if complete else "Insufficient folded-arm guard context around the nominated extension; no repair attempted."}


def survey(dataset_root: Path, output: Path) -> dict:
    output = output.resolve()
    output.relative_to((ROOT / "outputs/task10d/phase2").resolve())
    output.mkdir(parents=True, exist_ok=True)
    paths = [p for p in sorted((dataset_root / "fbx_export_joints").iterdir())
             if re.match(r"\d+_ma_(jab|cross|leadhook|rearhook)_liu_[^_]+_joints\.npy$", p.name)]
    counts = dict.fromkeys(EXPECTED, 0)
    recordings, events = [], []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(8, 2, figsize=(17, 21), constrained_layout=True)
    for path, ax in zip(paths, axes.flat):
        family = FAMILY[path.name.split("_")[2]]
        counts[family] += 1
        data, digest = read_recording(path)
        p = data["ma_position"]
        hand = 8 if family in ("jab", "left_hook") else 12
        reach = np.linalg.norm(p[:, hand]-p[:, hand-2], axis=1)
        peaks, _ = find_peaks(reach, distance=60, prominence=.12)
        local = []
        for i, peak in enumerate(peaks):
            lo = 0 if i == 0 else int((peaks[i-1]+peak)//2)
            hi = len(p)-1 if i == len(peaks)-1 else int((peaks[i+1]+peak)//2)
            row = event_row(path, data, digest, family, int(peak), lo, hi)
            if row["event_id"] in REVIEW_BOUNDS:
                row = event_row(path, data, digest, family, int(peak), lo, hi,
                                explicit_bounds=REVIEW_BOUNDS[row["event_id"]])
            if not row["hand_evidence_agrees_with_family"]:
                row.update(status="exclude_ambiguous_active_hand", reason="Initial recording fragment has larger opposite-arm reach excursion than the named family arm; do not infer anatomy from drill name.")
            if row["source_frame_range_inclusive"][0] < 8 or row["source_frame_range_inclusive"][1] > len(p)-9:
                row.update(status="exclude_recording_boundary", reason="Nominated crop abuts the recording boundary; full preparation or recovery context is uncertain.")
            if path.name.startswith("025_"):
                row.update(status="exclude_partner_clinch", reason="Jab-to-hug partner interaction; withdrawal depends on a clinch and offers no clean isolated return candidate without a new edit.")
            if row["event_id"] in DIVERSITY:
                row.update(status="selected_for_conversion_review", reason=DIVERSITY[row["event_id"]])
            local.append(row)
        events.extend(local)
        recordings.append({"filename": path.name, "path": str(path.resolve()), "sha256": digest,
                           "family": family, "frames": len(p), "duration_s": (len(p)-1)/FPS,
                           "actor": "ma", "partner_actor": "liu", "source_hz": FPS,
                           "nominated_reach_peak_count": len(local),
                           "complete_guard_context_candidate_count": sum(r["complete_guard_context_candidate"] for r in local),
                           "extension_frames": [int(i) for i in peaks]})
        ax.plot(np.arange(len(reach))/FPS, reach, lw=1)
        ax.scatter(peaks/FPS, reach[peaks], s=12, color="firebrick")
        for peak in peaks:
            ax.annotate(str(peak), (peak/FPS, reach[peak]), xytext=(0, 4), textcoords="offset points", fontsize=6, rotation=55)
        ax.set(title=path.stem, xlabel="recording time (s)", ylabel="wrist-to-shoulder (m)", ylim=(.10, .68))
    if counts != EXPECTED:
        raise ValueError(f"Focused inventory differs from reported counts: {counts}")
    fig.suptitle("Task 10D Phase 2: focused 16-recording source survey\nReach peaks nominate events; annotations are original 120 Hz frames, not contact times.")
    fig.savefig(output / "recording_overview.png", dpi=125)
    plt.close(fig)
    result = {"schema": "task10d_phase2_source_survey_v1", "inventory": counts,
              "scope": "Only sixteen named single-family recordings; no catalogue of the complete dataset.",
              "method": "Active wrist-to-shoulder reach peaks, prominence 0.12 m and separation >=0.5 s, nominate events. Folded-arm local minima bound recorded guard context. These descriptive thresholds are discovery aids, not biomechanical acceptance or exhaustive labels.",
              "nominated_reach_peak_count": len(events), "recordings": recordings, "events": events,
              "limitations": ["All sixteen recordings feature the same named attacker ma; partner-response contexts and within-performer execution variation do not establish broad population/style coverage.",
                              "Hand and stance are explicit source geometry; lead/rear drill names do not establish anatomy.",
                              "Survey peaks are extension descriptors, never nominal first-entry times.",
                              "Recorded guards may be low and may require explicit H2 morphology adaptation."]}
    (output / "survey.json").write_text(json.dumps(result, indent=2)+"\n")
    selected = [row for row in events if row["event_id"] in DIVERSITY]
    manifest = {"schema": "task10d_explicit_source_events_v1", "dataset_root": str(dataset_root.resolve()),
                "selection_status": "Source filmstrips inspected; convert only after the four nominal families are established. H2 suitability remains to be checked.",
                "source_recording_count": counts, "events": selected,
                "review_images": {family: str(output / f"{family}_source_review.png") for family in REVIEW_SELECTION},
                "retarget_config": str(ROOT / "configs/task10c/duobox_to_h2_31.json"),
                "processing": {"source_hz": 120, "output_hz": 50, "smoothing_sigma_source_frames": 1.5,
                               "sole_clearance_m": .002, "terminal_guard_hold_s": .4}}
    (output / "selected_events.json").write_text(json.dumps(manifest, indent=2)+"\n")
    with (output / "events.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(events[0]))
        writer.writeheader()
        writer.writerows(events)
    return result


def preconversion_checks(result: dict, dataset_root: Path, output: Path) -> None:
    """Check exact provenance/time/hand and describe endpoint velocity honestly."""
    selected = [r for r in result["events"] if r["event_id"] in DIVERSITY]
    checks = []
    for event in selected:
        clip = load_event(event, dataset_root)
        data, _ = read_recording(dataset_root / "fbx_export_joints" / event["source_recording"], event["source_sha256"])
        p = np.asarray(data["ma_position"], dtype=float)
        velocity = np.gradient(p, 1/120, axis=0)
        first, last = event["source_frame_range_inclusive"]
        hand = clip.active_hand_index
        checks.append({"event_id": event["event_id"], "source_bounds_inclusive": [first, last],
                       "frame_count": len(clip.time_s), "time_zero_frame": first,
                       "source_duration_s": clip.duration_s,
                       "expected_50hz_last_time_s": float(np.arange(0, clip.duration_s+1e-8, .02)[-1]),
                       "extension_time_s": (event["extension_frame"]-first)/120,
                       "source_sha_verified": True, "source_sha256": event["source_sha256"],
                       "source_fk_max_error_m": clip.metadata["fk_max_error_m"],
                       "anatomical_hand": event["anatomical_active_hand"],
                       "anatomy_evidence_left_right_reach_excursion_m": event["bilateral_wrist_shoulder_reach_excursion_m_left_right"],
                       "hand_evidence_agrees": event["hand_evidence_agrees_with_family"], "stance": event["stance"],
                       "source_root_endpoint_speed_m_s": np.linalg.norm(velocity[[first, last], 0], axis=-1).tolist(),
                       "source_active_wrist_endpoint_speed_m_s": np.linalg.norm(velocity[[first, last], hand], axis=-1).tolist(),
                       "max_source_step_root_m": float(np.linalg.norm(np.diff(p[first:last+1, 0], axis=0), axis=-1).max()),
                       "max_source_step_active_wrist_m": float(np.linalg.norm(np.diff(p[first:last+1, hand], axis=0), axis=-1).max()),
                       "endpoint_interpretation": "Reviewed recorded guard bounds include quiet source preparation/recovery. Source velocities are nonzero; H2 terminal settling and continuity to hold still require post-conversion review. No exact root/foot restoration is imposed."})
    summary = {"inventory": result["inventory"],
               "reach_peak_nominations_by_family": dict(Counter(x["family"] for x in result["events"])),
               "dispositions": dict(Counter(x["status"] for x in result["events"])),
               "additional_problem_notes": ["jab_011_00076 is also anatomically ambiguous by bilateral excursion and sits at the recording boundary.",
                                            "Original first-pass jab crop ended during recorded stepping. Reviewed source endpoint extended from frame1172 to1235 to include that step; no retargeting was needed to detect this."],
               "selected_source_checks": checks,
               "command": "source scripts/task10d_env.sh; .venv-task10c/bin/python -B -m human_punching.task10d.source_survey",
               "conversion": "Prepared separately; this script performs no conversion."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (output / "preconversion_checks.json").write_text(json.dumps({"schema": "task10d_source_event_preconversion_checks_v1", "checks": checks,
        "limitations": "Source reach extrema describe extension, not frozen-target first entry; native H2 derivative/endpoint checks follow conversion."}, indent=2)+"\n")


def source_reviews(result: dict, dataset_root: Path, output: Path) -> None:
    """Recorded-source filmstrips; these are not retargeted H2 previews."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    indexed = {row["event_id"]: row for row in result["events"]}
    for family, ids in REVIEW_SELECTION.items():
        fig = plt.figure(figsize=(17, 7), constrained_layout=True)
        for row_index, event_id in enumerate(ids):
            row = indexed[event_id]
            clip = load_event(row, dataset_root)
            p = clip.positions.copy()
            forward = clip.global_rotations[0, 0] @ np.array([0., 0., 1.])
            yaw = np.arctan2(forward[1], forward[0])
            c, s = np.cos(yaw), np.sin(yaw)
            rotate = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.]])
            origin = p[0, 0].copy(); origin[2] = 0
            p = (p-origin) @ rotate.T
            at = row["extension_frame"]-row["source_frame_range_inclusive"][0]
            frames = [0, at//2, at, (at+len(p)-1)//2, len(p)-1]
            hand = clip.active_hand_index
            center = (p.max(axis=(0, 1))+p.min(axis=(0, 1)))/2
            for column, frame in enumerate(frames):
                ax = fig.add_subplot(2, 5, row_index*5+column+1, projection="3d")
                for j, parent in enumerate(PARENTS[1:], 1):
                    color = "#c92a2a" if j in (hand-2, hand-1, hand) else "#315a82"
                    points = p[frame, [parent, j]]
                    ax.plot(*points.T, color=color, lw=2.6)
                ax.scatter(*p[frame, hand], color="#d00000", s=50)
                ax.plot(*p[:, hand].T, color="#d00000", alpha=.2, lw=1)
                ax.set(xlim=(center[0]-.8, center[0]+.8), ylim=(center[1]-.8, center[1]+.8), zlim=(0, 1.9))
                ax.set_box_aspect((1.6, 1.6, 1.9)); ax.view_init(elev=18, azim=-50)
                ax.set_title(f"{event_id}\nframe {clip.source_frames[frame]} / t={clip.time_s[frame]:.3f}s", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([0, 1])
        fig.suptitle(f"{family}: recorded source of nominal (top) and candidate variation (bottom)\nOriginal skeleton, source timing, red anatomical active arm; no H2 adaptation shown")
        fig.savefig(output / f"{family}_source_review.png", dpi=110)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "DuoBox")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/task10d/phase2/survey")
    args = parser.parse_args()
    result = survey(args.dataset_root.resolve(), args.output)
    preconversion_checks(result, args.dataset_root.resolve(), args.output.resolve())
    source_reviews(result, args.dataset_root.resolve(), args.output.resolve())
    print(json.dumps({"inventory": result["inventory"], "nominated_reach_peak_count": result["nominated_reach_peak_count"]}))


if __name__ == "__main__":
    main()
