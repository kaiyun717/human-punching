#!/usr/bin/env python3
"""Join Phase3 geometric placements to identity-matched native reset measurements.

Absent rendering is unknown. Counts refer to this TRAIN design subset and the
retained placement alternatives, never full-bank or universal impossibility.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
LAYERS = ("gross", "clean_id", "uncorrupted_c")
METRICS = ("glove_valid_source_pixels", "glove_rgb_valid_source_pixels", "glove_clean_cells",
           "glove_acquired_cells", "glove_clean_min_cells", "glove_acquired_min_cells")
FAMILIES = ("jab", "cross", "left_hook", "right_hook")
PASS, FAIL, UNKNOWN = 1, 0, -1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def family(primitive: str) -> str:
    return next(name for name in FAMILIES if primitive.startswith(name + "_"))


def quantiles(values) -> dict | None:
    values = np.asarray(values)
    if not values.size:
        return None
    return dict(zip(("min", "p05", "median", "p95", "max"),
                    np.quantile(values, [0, .05, .5, .95, 1]).tolist()))


def metric_quantiles(rows: list[dict]) -> dict:
    result = {"selected_measured_rows": len(rows)}
    for metric in METRICS:
        values = np.asarray([row["metrics"][metric] for row in rows])
        result[metric] = None if not len(values) else dict(
            left=quantiles(values[:, 0]), right=quantiles(values[:, 1]),
            worse_hand=quantiles(values.min(-1)))
    return result


def observation_flags(row: dict) -> tuple[bool, bool, bool]:
    """Only truthful same-timestamp, real source-pixel and union-cell evidence."""
    if row.get("atomic_preserved") is not True:
        raise ValueError("Atomic reset/launch preservation did not pass")
    metrics = row["metrics"]
    if (metrics["capture_timestamps"] != 0 or metrics["delivered_capture_timestamps"] != 0
            or metrics["padding"] is not True):
        raise ValueError("Native measurement changed reset timestamp or truthful padding")
    for key in METRICS:
        values = np.asarray(metrics[key])
        if values.shape != (2,) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"Invalid per-hand count: {key}")
    clean = bool(np.all(np.asarray(metrics["glove_valid_source_pixels"]) > 0)
                 and np.all(np.asarray(metrics["glove_clean_min_cells"]) > 0))
    rgb = bool(np.all(np.asarray(metrics["glove_rgb_valid_source_pixels"]) > 0)
               and np.all(np.asarray(metrics["glove_acquired_min_cells"]) > 0))
    if clean != row["clean_both_represented"] or rgb != row["rgb_both_represented"]:
        raise ValueError("Native summary flags disagree with measured source/cell counts")
    gross = row.get("gross_intersection")
    if type(gross) is not bool or row.get("threshold_m") != -.03:
        raise ValueError("Unexpected native gross-intersection convention")
    if gross != (row["minimum_native_convex_distance_capped_m"] < -.03):
        raise ValueError("Native gross-intersection flag disagrees with distance")
    return not gross, (not gross and clean), (not gross and rgb)


def aggregate_retained(geometric: np.ndarray, measured: dict[int, dict]) -> tuple[tuple[int, ...], int, str]:
    """Pass if a retained placement passes; otherwise expose any unknown ranks."""
    ranks = np.flatnonzero(geometric).tolist()
    status = []
    for layer in range(3):
        if any(rank in measured and measured[rank]["flags"][layer] for rank in ranks):
            status.append(PASS)
        elif any(rank not in measured for rank in ranks):
            status.append(UNKNOWN)
        else:
            status.append(FAIL)
    # C information is preferred, then ID oracle, then no gross intersection;
    # otherwise retain the geometric choice while exposing its failed/unknown layer.
    for layer, label in ((2, "uncorrupted_c"), (1, "clean_id"), (0, "gross_checked")):
        passing = [rank for rank in ranks if rank in measured and measured[rank]["flags"][layer]]
        if passing:
            return tuple(status), min(passing), label
    return tuple(status), (min(ranks) if ranks else -1), ("geometric_only" if ranks else "geometrically_unavailable")


def summarize_status(matrix: np.ndarray, primitive_ids: list[str]) -> dict:
    """Tri-state aggregation across alternatives, then primitive/family/state."""
    passed, unknown = matrix == PASS, matrix == UNKNOWN
    state_pass = passed.any(-1)
    state_unknown = ~state_pass & unknown.any(-1)
    family_pass = np.column_stack([passed[:, [family(pid) == f for pid in primitive_ids]].any(-1)
                                   for f in FAMILIES])
    family_unknown = np.column_stack([(~family_pass[:, index])
        & unknown[:, [family(pid) == f for pid in primitive_ids]].any(-1)
        for index, f in enumerate(FAMILIES)])
    per_primitive = {pid: dict(confirmed_pass=int(passed[:, i].sum()), unknown=int(unknown[:, i].sum()),
        no_passing_retained_placement=int((matrix[:, i] == FAIL).sum())) for i, pid in enumerate(primitive_ids)}
    return dict(state_count=len(matrix), states_with_confirmed_attack=int(state_pass.sum()),
        states_without_confirmed_attack_unknown=int(state_unknown.sum()),
        states_with_no_passing_retained_placement=int((~state_pass & ~state_unknown).sum()),
        state_primitive_pass_pairs=int(passed.sum()), state_primitive_unknown_pairs=int(unknown.sum()),
        per_primitive=per_primitive,
        per_family={f: dict(confirmed_available_states=int(family_pass[:, i].sum()),
            unknown_states=int(family_unknown[:, i].sum()),
            no_passing_retained_placement_states=int((~family_pass[:, i] & ~family_unknown[:, i]).sum()))
            for i, f in enumerate(FAMILIES)},
        confirmed_primitive_count_histogram=np.bincount(passed.sum(-1), minlength=len(primitive_ids)+1).tolist(),
        confirmed_family_count_histogram=np.bincount(family_pass.sum(-1), minlength=5).tolist(),
        unknown_family_count_histogram=np.bincount(family_unknown.sum(-1), minlength=5).tolist())


def summarize(search: Path, native_paths: list[Path], output: Path) -> dict:
    search, output = search.resolve(strict=True), output.resolve()
    if not output.is_relative_to((PROJECT / "outputs/task10d/phase3").resolve()):
        raise ValueError("Analysis outputs must remain within Phase3")
    if output.exists():
        raise FileExistsError("Use an explicitly new analysis directory")
    summary_path = search / "summary.json"
    original = json.loads(summary_path.read_text())
    geometry_path = Path(original["geometry_path"])
    if sha256(geometry_path) != original["geometry_sha256"]:
        raise ValueError("Current prepared geometry differs from search input")
    primitive_ids = original["primitive_ids"]
    if (len(set(primitive_ids)) != 8
            or any(sum(family(pid) == name for pid in primitive_ids) != 2 for name in FAMILIES)):
        raise ValueError("Phase3 summary requires all eight accepted primitives, two per family")
    pitches = np.asarray([row["pitch_up_deg"] for row in original["pitches"]], dtype=float)
    records, input_files = [], []
    ids = None
    for pid, name in zip(primitive_ids, original["candidate_files"], strict=True):
        path = search / name
        with np.load(path, allow_pickle=False) as source:
            record = {key: source[key].copy() for key in source.files}
        if ids is None:
            ids = record["snapshot_id"]
        np.testing.assert_array_equal(record["snapshot_id"], ids)
        np.testing.assert_array_equal(record["pitches_up_deg"], pitches)
        np.testing.assert_array_equal(record["available"].sum(-1),
            [row["per_primitive_available"][pid] for row in original["pitches"]])
        records.append(record)
        input_files.append(dict(path=str(path), sha256=sha256(path)))
    assert ids is not None
    if len(ids) != original["state_count"] or len(np.unique(ids)) != len(ids):
        raise ValueError("Search state count or original identities changed")
    sid_lookup = {int(sid): index for index, sid in enumerate(ids)}
    pid_lookup = {pid: index for index, pid in enumerate(primitive_ids)}
    pitch_lookup = {float(p): index for index, p in enumerate(pitches)}
    # Keys distinguish candidate alternatives and the literal accepted baseline.
    measurements: dict[tuple[int, int, int, int], dict] = {}
    ambiguous: set[tuple[int, int, int, int]] = set()
    errors, duplicate_count = [], 0
    for file_index, native_path in enumerate(native_paths):
        native_path = native_path.resolve(strict=True)
        invocation_path = native_path.parent / "invocation.json"
        invocation = json.loads(invocation_path.read_text())
        inline_hash = invocation.get("geometry_sha256")
        if (inline_hash is not None and inline_hash != original["geometry_sha256"]) or invocation.get("dynamic"):
            raise ValueError("Native input must be static measurements of the exact study geometry")
        if inline_hash is None:
            # The initial 88 nominal-baseline acquisitions predate inline hashes.
            # Accept only this supplemental nominal scope, retaining that gap.
            suffix = "/human-punching/" + str(geometry_path.relative_to(PROJECT))
            if not str(invocation["geometry"]).endswith(suffix):
                raise ValueError("Unhashed nominal baseline does not name this geometry path")
        file_record = dict(path=str(native_path), sha256=sha256(native_path),
            invocation_sha256=sha256(invocation_path), job_id=invocation["job_id"], parsed_rows=0,
            native_inline_geometry_sha256=inline_hash,
            geometry_identity_caveat=None if inline_hash is not None else
                "Inline geometry hash was not recorded by this earlier invocation. Accepted only as supplemental nominal accepted-baseline evidence by original path, unchanged current geometry identity and existing native reset assertions; no historical inline hash is fabricated.")
        for line_number, line in enumerate(native_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            file_record["parsed_rows"] += 1
            try:
                p, s, j = pitch_lookup[float(row["pitch_up_deg"])], sid_lookup[int(row["snapshot_id"])], pid_lookup[row["primitive_id"]]
                rank = int(row["rank"])
                record = records[j]
                if inline_hash is None and not (rank == -1 and row["kind"] == "accepted_baseline"
                                                and row["snapshot_id"] == original["nominal_snapshot_id"]):
                    raise ValueError("Missing-inline-hash input is restricted to supplemental nominal baseline")
                if rank == -1:
                    if row["kind"] != "accepted_baseline":
                        raise ValueError("Baseline rank must name accepted_baseline")
                    expected_xy, expected_yaw = np.zeros(2), 0.
                    geometric = bool(record["baseline_joint"][p, s])
                else:
                    if row["kind"] != "bounded_candidate" or not 0 <= rank < record["top_available"].shape[-1]:
                        raise ValueError("Candidate rank/kind does not name a retained search placement")
                    expected_xy, expected_yaw = record["top_xy"][p, s, rank], record["top_yaw_rad"][p, s, rank]
                    geometric = bool(record["top_available"][p, s, rank])
                    if not geometric:
                        raise ValueError("Rendered candidate was not geometrically admitted")
                np.testing.assert_allclose(row["placement"]["xy"], expected_xy, atol=1e-8, rtol=0)
                angle_error = np.arctan2(np.sin(row["placement"]["yaw"] - expected_yaw),
                                         np.cos(row["placement"]["yaw"] - expected_yaw))
                if abs(angle_error) > 1e-8:
                    raise ValueError("Native yaw differs from the same retained search placement")
                flags = observation_flags(row)
                # No native observation can confer reach/frustum on a baseline that fails geometry.
                flags = tuple(bool(geometric and value) for value in flags)
                key = p, s, j, rank
                value = dict(row=row, flags=flags, geometric=geometric, file_index=file_index,
                             line_number=line_number)
                if key in measurements:
                    duplicate_count += 1
                    if measurements[key]["flags"] != flags:
                        ambiguous.add(key)
                else:
                    measurements[key] = value
            except (KeyError, ValueError, AssertionError) as error:
                errors.append(dict(path=str(native_path), line=line_number, error=str(error)[:600],
                    snapshot_id=row.get("snapshot_id"), primitive_id=row.get("primitive_id"),
                    pitch_up_deg=row.get("pitch_up_deg"), rank=row.get("rank")))
        input_files.append(file_record)
    for key in ambiguous:
        measurements.pop(key, None)
    shape = len(pitches), len(ids), len(primitive_ids)
    statuses = np.full((*shape, 3), UNKNOWN, np.int8)
    geometry = np.stack([r["available"] for r in records], -1)
    baseline_geometry = np.stack([r["baseline_joint"] for r in records], -1)
    baseline_status = np.where(baseline_geometry[..., None], UNKNOWN, FAIL).repeat(3, axis=-1).astype(np.int8)
    selected_rank = np.full(shape, -1, np.int8)
    selected_class = np.zeros(shape, np.int8)
    selected_file = np.full(shape, -1, np.int16)
    selected_line = np.full(shape, -1, np.int32)
    selected_xy = np.full((*shape, 2), np.nan)
    selected_yaw = np.full(shape, np.nan)
    classes = {"geometrically_unavailable": 0, "geometric_only": 1, "gross_checked": 2, "clean_id": 3, "uncorrupted_c": 4}
    selected_rows: dict[tuple[int, int, int], dict] = {}
    for p in range(len(pitches)):
        for j, record in enumerate(records):
            for s in range(len(ids)):
                measured = {k: measurements[p, s, j, k] for k in range(record["top_available"].shape[-1])
                            if (p, s, j, k) in measurements}
                status, rank, category = aggregate_retained(record["top_available"][p, s], measured)
                statuses[p, s, j] = status
                selected_rank[p, s, j] = rank
                selected_class[p, s, j] = classes[category]
                if rank >= 0:
                    selected_xy[p, s, j] = record["top_xy"][p, s, rank]
                    selected_yaw[p, s, j] = record["top_yaw_rad"][p, s, rank]
                if rank in measured:
                    selected_rows[p, s, j] = measured[rank]["row"]
                    selected_file[p, s, j] = measured[rank]["file_index"]
                    selected_line[p, s, j] = measured[rank]["line_number"]
                baseline = measurements.get((p, s, j, -1))
                if baseline:
                    baseline_status[p, s, j] = [PASS if value else FAIL for value in baseline["flags"]]
    nominal_index = int(np.flatnonzero(ids == original["nominal_snapshot_id"])[0])
    by_pitch = []
    for p, pitch in enumerate(pitches):
        nominal = {}
        rows = []
        for j, pid in enumerate(primitive_ids):
            row = selected_rows.get((p, nominal_index, j))
            nominal[pid] = dict(geometric=bool(geometry[p, nominal_index, j]),
                selected_rank=int(selected_rank[p, nominal_index, j]),
                statuses={layer: int(statuses[p, nominal_index, j, index]) for index, layer in enumerate(LAYERS)},
                metrics=None if row is None else row["metrics"],
                gross_intersection=None if row is None else row["gross_intersection"],
                baseline_measurement=None if (p, nominal_index, j, -1) not in measurements
                    else measurements[p, nominal_index, j, -1]["row"])
        rows = [row for (p0, _, _), row in selected_rows.items() if p0 == p]
        native_keys = [key for key in measurements if key[0] == p and key[3] >= 0]
        row_summary = dict(pitch_up_deg=float(pitch),
            geometric=summarize_status(np.where(geometry[p], PASS, FAIL), primitive_ids),
            baseline_geometric=summarize_status(np.where(baseline_geometry[p], PASS, FAIL), primitive_ids),
            native_measured_candidate_placements=len(native_keys),
            native_measured_state_primitive_pairs=len({key[:3] for key in native_keys}),
            native_intersection_failures=sum(measurements[key]["row"]["gross_intersection"] for key in native_keys),
            nominal=nominal, selected_measurement_quantiles=metric_quantiles(rows),
            selected_measurement_quantiles_per_primitive={pid: metric_quantiles(
                [row for (p0, _, j0), row in selected_rows.items() if p0 == p and j0 == j])
                for j, pid in enumerate(primitive_ids)},
            selected_rank_counts=dict(Counter(map(str, selected_rank[p][selected_rank[p] >= 0].ravel().tolist()))))
        for index, layer in enumerate(LAYERS):
            row_summary[layer] = summarize_status(statuses[p, ..., index], primitive_ids)
            row_summary["baseline_" + layer] = summarize_status(baseline_status[p, ..., index], primitive_ids)
        by_pitch.append(row_summary)
    output.mkdir(parents=True)
    payload = output / "selected_placements.npz"
    np.savez_compressed(payload, snapshot_id=ids, primitive_id=np.asarray(primitive_ids), pitch_up_deg=pitches,
        geometric=geometry, native_layer_status=statuses, baseline_native_layer_status=baseline_status,
        selected_rank=selected_rank, selected_class=selected_class, selected_xy=selected_xy,
        selected_yaw_rad=selected_yaw, selected_measurement_file=selected_file,
        selected_measurement_line=selected_line)
    result = dict(schema="task10d_phase3_verified_subset_summary_v1", state_count=len(ids),
        scope="TRAIN posture-covering design subset only; no full-bank coverage run",
        search_path=str(search), search_summary_sha256=sha256(summary_path),
        input_files=input_files, native_measurement_paths=[str(p.resolve()) for p in native_paths],
        selected_placements_sha256=sha256(payload), selection_classes=classes,
        selected_measurement_reference="selected_measurement_file indexes native_measurement_paths; selected_measurement_line is one-based JSONL line; -1 means no valid selected native row",
        status_codes={"confirmed_pass": PASS, "no_passing_retained_placement": FAIL, "unknown": UNKNOWN},
        layers={"geometric": "Sampled active-outward 0.14m reach, both initial 0.16m clearances, same-placement center frustum and declared candidate frontal bounds",
            "gross": "Geometric plus same-placement native reset measurement and no cross-character convex penetration deeper than 3cm",
            "clean_id": "Gross layer plus each glove supplies valid source depth pixels and >=1 minimum-contributing cell of unchanged clean ID-union 16x9 reduction",
            "uncorrupted_c": "Gross layer plus each glove supplies valid RGB-HSV source pixels and >=1 minimum-contributing cell of unchanged uncorrupted C-union 16x9 reduction"},
        selection_rule="Lowest ranked retained placement passing uncorrupted C, then clean ID, then gross; otherwise lowest geometric rank. No unmeasured observation is treated as passed.",
        unknown_rule="An available retained placement with no valid native result remains unknown. A failed tested placement does not fail unrendered alternatives. A no-pass result after all retained alternatives is scoped to these alternatives, not universal geometric/visual impossibility.",
        duplicates=duplicate_count, ambiguous_measurement_keys=[list(key) for key in sorted(ambiguous)],
        data_reset_transform_errors=errors, pitches=by_pitch,
        render_incomplete=bool(np.any(statuses == UNKNOWN)), script_sha256=sha256(Path(__file__)))
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["| Pitch up | Geometric states | Native gross-pass states | Clean-ID states | Uncorrupted-C states | C unknown states | Measured placements |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in by_pitch:
        lines.append(f"| {row['pitch_up_deg']:+g}° | {row['geometric']['states_with_confirmed_attack']} | "
            f"{row['gross']['states_with_confirmed_attack']} | {row['clean_id']['states_with_confirmed_attack']} | "
            f"{row['uncorrupted_c']['states_with_confirmed_attack']} | {row['uncorrupted_c']['states_without_confirmed_attack_unknown']} | "
            f"{row['native_measured_candidate_placements']} |")
    lines.extend(["", "Per-hand source/cell quantiles use the worse hand of each selected measured placement; failed observations contribute zero.", "",
        "| Pitch up | Selected measured pairs | C valid source pixels min / p05 / median | C minimum-contributing cells min / p05 / median | ID minimum-contributing cells min / p05 / median |",
        "|---:|---:|---:|---:|---:|"])
    for row in by_pitch:
        metric = row["selected_measurement_quantiles"]
        if not metric["selected_measured_rows"]:
            continue
        def triple(name):
            values = metric[name]["worse_hand"]
            return " / ".join(f"{values[key]:g}" for key in ("min", "p05", "median"))
        lines.append(f"| {row['pitch_up_deg']:+g}° | {metric['selected_measured_rows']} | "
            f"{triple('glove_rgb_valid_source_pixels')} | {triple('glove_acquired_min_cells')} | "
            f"{triple('glove_clean_min_cells')} |")
    (output / "table.txt").write_text("\n".join(lines) + "\n")
    return result


def self_test() -> None:
    geometric = np.array([True, True, True, False])
    # Failed first placement does not turn its unrendered alternatives into failures.
    measured = {0: {"flags": (True, False, False)}}
    status, rank, label = aggregate_retained(geometric, measured)
    assert status == (PASS, UNKNOWN, UNKNOWN) and rank == 0 and label == "gross_checked"
    # C on a later alternative takes precedence over earlier oracle-only evidence.
    measured[0] = {"flags": (True, True, False)}
    measured[2] = {"flags": (True, True, True)}
    status, rank, label = aggregate_retained(geometric, measured)
    assert status == (PASS, PASS, PASS) and rank == 2 and label == "uncorrupted_c"
    # A known geometrical exclusion cannot be upgraded by missing render data.
    assert aggregate_retained(np.zeros(4, bool), {})[0] == (FAIL, FAIL, FAIL)
    matrix = np.array([[FAIL, UNKNOWN], [PASS, UNKNOWN], [FAIL, FAIL]])
    result = summarize_status(matrix, ["jab_nominal", "jab_004_01107"])
    assert result["states_with_confirmed_attack"] == 1
    assert result["states_without_confirmed_attack_unknown"] == 1
    assert result["states_with_no_passing_retained_placement"] == 1
    assert result["confirmed_family_count_histogram"] == [2, 1, 0, 0, 0]
    print("Four incomplete-render/alternative/family aggregation checks passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path, default=PROJECT / "outputs/task10d/phase3/search01")
    parser.add_argument("--measurements", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.measurements or args.output is None:
        parser.error("--measurements and --output are required")
    result = summarize(args.search, args.measurements, args.output)
    print(json.dumps(dict(output=str(args.output.resolve()),
        native_rows=sum(record.get("parsed_rows", 0) for record in result["input_files"]),
        ambiguous=len(result["ambiguous_measurement_keys"]),
        data_errors=len(result["data_reset_transform_errors"])), indent=2))


if __name__ == "__main__":
    main()
