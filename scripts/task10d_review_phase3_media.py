"""Focused posthoc review of completed Phase 3 source-clock previews.

No simulation, rendering, new acquisition, or alteration of the source run.
Video sample counts use container tables; only selected review frames are decoded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from human_punching.task10d.g1 import PROJECT
from human_punching.task10d.phase2 import load_library
from human_punching.task10d.placement_phase3 import INDEX, PRIMITIVE_IDS
from human_punching.task10d.replay import Placement


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_time(times, mask):
    found = np.flatnonzero(mask)
    return None if not len(found) else float(times[found[0]])


def same_event(actual, expected, label):
    if actual is None or expected is None:
        if actual != expected:
            raise AssertionError(f"{label}: {actual} != {expected}")
    else:
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0, err_msg=label)


def video_sample_count(path):
    """Read the selected video track's MP4 stsz sample count, without decoding."""
    data = Path(path).read_bytes()

    def boxes(start, stop):
        while start < stop:
            if start+8 > stop:
                raise ValueError("Truncated MP4 box header")
            size, kind = struct.unpack_from(">I4s", data, start)
            header = 8
            if size == 1:
                size = struct.unpack_from(">Q", data, start+8)[0]
                header = 16
            elif size == 0:
                size = stop-start
            if size < header or start+size > stop:
                raise ValueError("Invalid MP4 box extent")
            yield kind, start+header, start+size
            start += size

    def descendants(start, stop):
        for kind, first, last in boxes(start, stop):
            yield kind, first, last
            if kind in (b"moov", b"trak", b"mdia", b"minf", b"stbl"):
                yield from descendants(first, last)

    counts = []
    for kind, first, last in descendants(0, len(data)):
        if kind != b"trak":
            continue
        fields = list(descendants(first, last))
        if any(k == b"hdlr" and data[a+8:a+12] == b"vide" for k, a, _ in fields):
            counts.extend(struct.unpack_from(">I", data, a+8)[0]
                          for k, a, _ in fields if k == b"stsz")
    if len(counts) != 1:
        raise ValueError(f"Expected one nonfragmented MP4 video sample table: {path}")
    return int(counts[0])


def inspect_scenario(directory, request, primitive, expected_qvel_norm):
    summary = json.loads((directory / "summary.json").read_text())
    camera = json.loads((directory / "camera.json").read_text())
    with np.load(directory / "trace.npz", allow_pickle=False) as source:
        trace = source["values"].copy()
        columns = source["columns"].tolist()
    expected_columns = ['time','head_x','head_y','head_z','left_x','left_y','left_z',
        'right_x','right_y','right_z','frozen_left','frozen_right','moving_left',
        'moving_right','pelvis_x','pelvis_y','pelvis_z']
    assert columns == expected_columns and np.isfinite(trace).all()
    assert summary["clock"] == "source" and summary["diagnostic_continuation"]
    assert summary["safe_arrival_evaluated"] is False
    assert summary["primitive_id"] == request["primitive_id"]
    assert summary["snapshot_id"] == request["snapshot_id"]
    assert all(summary["atomic_unchanged"].values())
    reset = summary["reset"]
    assert reset["physics_steps"] == 0
    assert all(value == 0 for value in reset["restored_max_abs_error"].values())
    np.testing.assert_allclose(reset["transformed_qvel_norm"][0], expected_qvel_norm,
                               atol=2e-6, rtol=0)
    assert reset["transformed_qvel_norm"][0] > 0
    placement = Placement(**request["placement"])
    np.testing.assert_array_equal(summary["placement"]["xy"], placement.xy)
    assert summary["placement"]["yaw"] == placement.yaw
    times, head = trace[:, 0], trace[:, 1:4]
    gloves = trace[:, 4:10].reshape(-1, 2, 3)
    frozen = np.asarray(summary["frozen_target"])
    np.testing.assert_array_equal(frozen, head[0])
    np.testing.assert_allclose(times, np.arange(len(times)) * .005, atol=1e-12, rtol=0)
    expected_steps = 4 * int(np.ceil((primitive.reference.end_time_s + .4) / .02))
    assert len(times) == expected_steps + 1
    assert summary["frames"] == expected_steps // 4 + 1 == len(camera)
    np.testing.assert_allclose(summary["full_cycle_end_s"], times[-1], atol=1e-12, rtol=0)
    # Independent native FK from the one committed placement/source clock at
    # every saved 5 ms boundary; no alternate or moving-target reference.
    expected_gloves = np.stack([primitive.reference.sample(t, placement).glove_pos for t in times])
    path_error = float(np.max(np.abs(gloves - expected_gloves)))
    np.testing.assert_allclose(gloves, expected_gloves, atol=5e-6, rtol=0)
    # The runtime computes distances from native float32 arrays before saving
    # the mixed trace as float64. Reproduce that arithmetic at the same 0.14/
    # 0.16 thresholds rather than introducing double-rounding event changes.
    frozen_dist = np.linalg.norm(gloves.astype(np.float32) - frozen.astype(np.float32), axis=-1)
    moving_dist = np.linalg.norm(gloves.astype(np.float32) - head[:, None].astype(np.float32), axis=-1)
    np.testing.assert_allclose(trace[:, 10:12], frozen_dist, atol=1e-12, rtol=0)
    np.testing.assert_allclose(trace[:, 12:14], moving_dist, atol=1e-12, rtol=0)
    first_frozen = [first_time(times, frozen_dist[:, hand] <= .14) for hand in range(2)]
    first_moving = [first_time(times, moving_dist[:, hand] <= .14) for hand in range(2)]
    first_clearance = first_time(times, moving_dist.min(-1) < .16)
    for hand in range(2):
        same_event(first_frozen[hand], summary["frozen_first_by_hand_s"][hand], "frozen event")
        same_event(first_moving[hand], summary["moving_first_by_hand_s"][hand], "moving event")
    same_event(first_clearance, summary["first_clearance_violation_s"], "clearance event")
    active = ("left", "right").index(primitive.active_hand)
    outward_end = float(primitive.record["annotations"]["withdrawal_start_s"])
    outward = times <= outward_end + 1e-12
    first_active = first_time(times, outward & (frozen_dist[:, active] <= .14))
    assert first_active is not None
    earlier_other = first_time(times, (times < first_active) & (frozen_dist[:, 1-active] <= .14))
    delay = int(camera[0]["delay_ticks"][0])
    assert delay in (2, 3, 4)
    delivered_counts, padded = [], []
    for frame, row in enumerate(camera):
        acquisition = .02 * frame
        source_frame = max(0, frame-delay)
        for key in ("capture_timestamps", "delivery_timestamps"):
            np.testing.assert_allclose(row[key], [acquisition], atol=1e-12, rtol=0)
        np.testing.assert_allclose(row["delivered_capture_timestamps"], [.02*source_frame], atol=1e-12, rtol=0)
        assert row["delay_ticks"] == [delay]
        assert row["padding"] == [frame < delay]
        delivered = np.asarray(row["delivered_masks"], dtype=bool)
        assert delivered.shape == (1, 9, 16)
        np.testing.assert_array_equal(delivered, camera[source_frame]["corrupted_masks"])
        values = np.asarray(row["delivered_normalized"])
        assert values.shape == delivered.shape and np.isfinite(values).all()
        assert ((values >= 0) & (values <= 1)).all()
        assert (values[~delivered] == 1).all()
        delivered_counts.append(int(delivered.sum()))
        padded.append(frame < delay)
    frame_times = np.arange(len(camera))*.02
    # When no moving clearance event occurs, stop this information statistic
    # at the outward boundary; terminal hold cannot improve its result.
    cutoff = min(outward_end, first_clearance if first_clearance is not None else outward_end)
    included = frame_times < cutoff - 1e-12
    counts = np.asarray(delivered_counts)
    pad = np.asarray(padded)
    pre = counts[included]
    sources = np.maximum(np.arange(len(camera))-delay, 0)
    prehazard = dict(cutoff_s=cutoff, cutoff="min(moving-head first 0.16m clearance event, outward end)",
        camera_frames=int(included.sum()), padded_frames=int((included & pad).sum()),
        distinct_actual_capture_frames=int(len(np.unique(sources[included]))),
        frames_with_delivered_valid_union=int((included & (counts > 0)).sum()),
        nonpadding_frames_with_delivered_valid_union=int((included & ~pad & (counts > 0)).sum()),
        delivered_union_cell_min=int(pre.min()) if len(pre) else None,
        delivered_union_cell_median=float(np.median(pre)) if len(pre) else None,
        delivered_union_cell_max=int(pre.max()) if len(pre) else None,
        first_nonpadding_informative_delivery_s=first_time(frame_times, included & ~pad & (counts > 0)),
        scope="ordinary-corrupted delayed union only; no delivered per-hand or actor-use claim")
    video_counts = {name: video_sample_count(directory / f"{name}.mp4")
                    for name in ("external", "perception")}
    assert all(value == len(camera) for value in video_counts.values())
    return dict(label=request["label"], snapshot_id=request["snapshot_id"],
        primitive_id=primitive.primitive_id, family=primitive.family,
        pitch_up_deg=request["pitch_up_deg"], kind=request["kind"],
        atomic_state_preservation_reported=True, independent_fixed_path_max_abs_error_m=path_error,
        transformed_qvel_norm=reset["transformed_qvel_norm"][0],
        source_active_outward_first_sample_s=first_active,
        source_active_outward_min_m=float(frozen_dist[outward, active].min()),
        earlier_inactive_frozen_entry_s=earlier_other,
        frozen_first_by_hand_s=first_frozen, moving_first_by_hand_s=first_moving,
        moving_head_active_encounter=first_moving[active] is not None,
        first_moving_clearance_violation_s=first_clearance,
        g1_max_head_displacement_m=float(np.linalg.norm(head-head[0], axis=-1).max()),
        g1_max_pelvis_displacement_m=float(np.linalg.norm(trace[:, -3:]-trace[0, -3:], axis=-1).max()),
        source_recovery_complete_s=primitive.reference.end_time_s,
        logged_end_s=float(times[-1]), sampled_physics_states=len(times), camera_frames=len(camera),
        video_sample_counts=video_counts, delay_ms=20*delay,
        initial_glove_valid_source_pixels=camera[0]["glove_valid_source_pixels"][0],
        initial_glove_id_min_cells=camera[0]["glove_clean_min_cells"][0],
        initial_glove_rgb_min_cells=camera[0]["glove_acquired_min_cells"][0],
        prehazard_delivered_union=prehazard, verified=True)


def image_sheet(path, items, *, columns=2, width=1008, height=450):
    sheet = Image.new("RGB", (columns*width, ((len(items)+columns-1)//columns)*height), "#151b23")
    draw = ImageDraw.Draw(sheet)
    for i, (label, picture) in enumerate(items):
        image = picture.copy()
        image.thumbnail((width, height-24))
        x, y = (i % columns)*width, (i // columns)*height
        draw.text((x+6, y+5), label, fill="white")
        sheet.paste(image, (x, y+24))
    sheet.save(path)


def selected_visuals(run, output, requests, records):
    by_label = {record["label"]: record for record in records}
    comparisons = [
        "baseline_jab_nominal_s53874_p-15", "nominal_jab_nominal_s53874_p25",
        "baseline_cross_012_01441_s53874_p25", "nominal_cross_012_01441_s53874_p25",
        "baseline_jab_004_01107_s53874_p25", "nominal_jab_004_01107_s53874_p25"]
    difficult = [r["label"] for r in requests if any(r["label"].startswith(prefix)
        for prefix in ("low_head_", "high_head_", "torso_roll_", "tight_margin_"))]
    for name, labels in (("initial_comparisons.jpg", comparisons), ("difficult_initial_scenes.jpg", difficult)):
        items = []
        for label in labels:
            with Image.open(run / label / "initial_scene.png") as source:
                items.append((label, source.copy()))
        image_sheet(output/name, items)
    # Decode two exact camera ticks in four selected complete videos, retaining
    # both external and perception panels. This is a bounded sample, not a
    # claim that every encoded frame received human visual review.
    selected = [comparisons[3], comparisons[5], difficult[0], difficult[-1]]
    decoded = []
    for label in selected:
        record = by_label[label]
        indices = sorted(set((int(np.floor(record["source_active_outward_first_sample_s"]/.02)),
                              int(np.floor(record["source_recovery_complete_s"]/.02)))))
        items = []
        for name in ("external", "perception"):
            with imageio.get_reader(run/label/f"{name}.mp4") as reader:
                for index in indices:
                    frame = reader.get_data(index)
                    items.append((f"{name} acquisition tick {index}, t={index*.02:.3f} s",
                                  Image.fromarray(frame)))
                    decoded.append(dict(label=label, video=name, frame=index, time_s=index*.02))
        image_sheet(output/f"decoded_{label}.jpg", items, width=1008, height=660)
    return dict(initial_comparisons=comparisons, difficult_initial_scenes=difficult,
                decoded_frames=decoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    boundary = (PROJECT / "outputs/task10d/phase3").resolve()
    if not run.is_relative_to(boundary) or not output.is_relative_to(boundary):
        raise ValueError("Review inputs/output must remain in Phase 3")
    completion = json.loads((run/"summary.json").read_text())
    requests = json.loads(args.requests.read_text())["requests"]
    assert completion["completed"] == len(requests)
    invocation = json.loads((run/"invocation.json").read_text())
    assert digest(args.requests) == invocation["requests_sha256"]
    assert digest(args.geometry) == invocation["geometry_sha256"]
    with np.load(args.geometry, allow_pickle=False) as source:
        qvel_norms = dict(zip(source["snapshot_id"].tolist(),
                             np.linalg.norm(source["sim/data/qvel"], axis=-1).tolist()))
    primitives = {p.primitive_id: p for p in load_library(INDEX, list(PRIMITIVE_IDS))}
    output.mkdir(parents=True, exist_ok=False)
    records, failures = [], []
    for request in requests:
        label = request["label"]
        try:
            result = inspect_scenario(run/label, request, primitives[request["primitive_id"]],
                                      qvel_norms[request["snapshot_id"]])
            records.append(result)
            print(f"PASS {label}: moving={result['moving_first_by_hand_s']}", flush=True)
        except Exception as error:
            failures.append(dict(label=label, type=type(error).__name__, detail=str(error)))
            print(f"FAIL {label}: {type(error).__name__}: {error}", flush=True)
    visuals = selected_visuals(run, output, requests, records) if not failures else None
    result = dict(schema="task10d_phase3_focused_dynamic_review_v1", source_run=str(run),
        job_id=completion["job_id"], requested=len(requests), verified=len(records), failures=failures,
        passed=not failures, records=records, selected_visuals=visuals,
        aggregate=dict(total_video_samples=sum(sum(r["video_sample_counts"].values()) for r in records),
            active_outward_entries=len(records), moving_active_encounters=sum(r["moving_head_active_encounter"] for r in records),
            earlier_inactive_frozen_entries=sum(r["earlier_inactive_frozen_entry_s"] is not None for r in records),
            max_reference_glove_error_m=max((r["independent_fixed_path_max_abs_error_m"] for r in records), default=None)),
        limitations=["Selected diagnostic scenarios, not a performance or generalization estimate.",
            "Reset preservation is the recorded native assertion, supplemented by independent trace/path and timestamp checks.",
            "Stored physical samples are 5 ms; no sub-step or continuous collision claim.",
            "Delivered information is the actual corrupted union; acquired per-hand counts cannot establish delayed per-hand visibility.",
            "Fresh96 is proprioceptive. These delayed visual packets did not enter its actor observation.",
            "Video counts use MP4 sample tables; only the listed 16 selected video frames were decoded for this review.",
            "Continued playback after first hazard is labeled diagnostic; no safe-arrival trial or policy acceptance."])
    (output/"review.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    lines = ["# Phase 3 focused dynamic replay review", "", f"Verified {len(records)}/{len(requests)} completed scenarios; failures: {len(failures)}.", "",
        "| Scenario | Frozen active entry | Moving active entry | First clearance | Pre-hazard valid delivered frames |",
        "|---|---:|---:|---:|---:|"]
    for record in records:
        active = ("left", "right").index(primitives[record["primitive_id"]].active_hand)
        info = record["prehazard_delivered_union"]
        lines.append(f"| {record['label']} | {record['source_active_outward_first_sample_s']:.3f} | {record['moving_first_by_hand_s'][active]} | {record['first_moving_clearance_violation_s']} | {info['frames_with_delivered_valid_union']}/{info['camera_frames']} |")
    lines += ["", *result["limitations"], ""]
    (output/"README.md").write_text("\n".join(lines))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
