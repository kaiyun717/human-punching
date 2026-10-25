"""Validate and summarize completed matched Task08/H2 diagnostic evaluations.

Reads existing records only. Full-source entrance times come from the immutable
episode input arrays; runtime events remain the independently observed first
terminal event. Media selection is illustrative, not a performance sample.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .task08_diagnostic_media import select_representatives

PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT / "outputs/task10d/task08_diagnostic_20261024"
DISPOSITIONS = ("arrival", "fist_clearance", "self_collision", "joint_limit", "height",
                "return_plus_3s_cutoff", "scheduler_watchdog")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify_existing(path, text):
    """Permit identical existing evidence without replacing it."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"Output artifacts must not be symlinks: {path}")
    if path.exists():
        existing = path.read_text()
        same = (json.loads(existing) == json.loads(text)) if path.suffix == ".json" else existing == text
        if not same:
            raise FileExistsError(f"Existing artifact differs; use a fresh output: {path}")


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return dict(count=0, min=None, median=None, max=None)
    if not np.isfinite(values).all():
        raise ValueError("Timing summaries require finite observed values")
    return dict(count=len(values), min=float(values.min()),
                median=float(np.median(values)), max=float(values.max()))


def describe(rows, inputs, input_lookup):
    indices = [input_lookup[int(row["episode_id"])] for row in rows]
    arrivals = [row for row in rows if row["safe_arrival"]]
    by_delay = {}
    for ticks in (2, 3, 4):
        group = [row for row in rows if row["delay_ticks"] == ticks]
        passed = sum(row["safe_arrival"] for row in group)
        by_delay[str(20 * ticks)] = dict(episodes=len(group), safe_arrivals=passed,
            safe_arrival_rate=None if not group else passed / len(group),
            dispositions={name: sum(row["disposition"] == name for row in group)
                          for name in DISPOSITIONS})
    recovery = [row["event_time_s"] - row["punch_return_time_s"] for row in arrivals]
    return dict(episodes=len(rows), safe_arrivals=len(arrivals),
        safe_arrival_rate=None if not rows else len(arrivals)/len(rows),
        dispositions={name: sum(row["disposition"] == name for row in rows) for name in DISPOSITIONS},
        initial_clean_both_count=sum(row["initial_clean_both"] for row in rows),
        physical_hit_count=sum(row["observed_hit"] for row in rows),
        by_delay_ms=by_delay,
        timing=dict(source_outward_entry_sampled_s=distribution(inputs["source_outward_entry_sample_s"][indices]),
            source_outward_entry_interpolated_s=distribution(inputs["source_outward_entry_interpolated_s"][indices]),
            source_recovery_complete_s=distribution(inputs["source_recovery_complete_s"][indices]),
            runtime_first_event_s=distribution([row["event_time_s"] for row in rows]),
            observed_safe_arrival_s=distribution([row["event_time_s"] for row in arrivals]),
            observed_safe_arrival_after_reference_return_s=distribution(recovery)))


def validate_rows(rows, inputs, manifest):
    """Reject missing, duplicate, reordered, mismatched or unfinished attempts."""
    primitives = manifest["primitive_ids"]
    if len(primitives) != 8 or len(set(primitives)) != 8 or manifest["episode_count"] != 8000:
        raise ValueError("Expected exactly eight primitives and8000 frozen episode inputs")
    ids = inputs["episode_id"]
    if len(ids) != 8000 or len(np.unique(ids)) != 8000 or set(ids.tolist()) != set(range(8000)):
        raise ValueError("Input episode IDs must be exactly0..7999")
    input_lookup = {int(value): i for i, value in enumerate(ids)}
    if len(rows) != 8000 or Counter(int(row["episode_id"]) for row in rows) != Counter(input_lookup.keys()):
        raise ValueError("Require exactly8000 unique, identity-matched completed episodes")
    for row in rows:
        i = input_lookup[int(row["episode_id"])]
        p = int(inputs["primitive_index"][i])
        expected = dict(primitive_id=primitives[p], snapshot_id=int(inputs["snapshot_id"][i]),
            snapshot_index=int(inputs["snapshot_index"][i]), camera_pitch_up_deg=25., source_clock_rate=1.)
        if any(row.get(name) != value for name, value in expected.items()):
            raise ValueError(f"Episode identity differs: {row['episode_id']}")
        if (row.get("completed") is not True or row.get("disposition") not in DISPOSITIONS
                or row.get("fatal", False) is not False or row.get("nonfinite", False) is not False):
            raise ValueError(f"Incomplete, numerical or unsupported outcome: {row['episode_id']}")
        for field in ("safe_arrival", "initial_clean_both", "observed_hit"):
            if type(row.get(field)) is not bool:
                raise ValueError(f"Missing Boolean outcome {field}")
        if row["safe_arrival"] != (row["disposition"] == "arrival"):
            raise ValueError("Arrival flag and first-event disposition disagree")
        if type(row.get("delay_ticks")) is not int or row["delay_ticks"] not in (2, 3, 4):
            raise ValueError("Expected ordinary episode-fixed40/60/80ms delay")
        event = row.get("event_time_s")
        if event is None or not np.isfinite(event) or event < 0 or event > 30. + 1e-8:
            raise ValueError("Completed episode requires a finite bounded first-event timestamp")
        for field in ("min_safe_clearance_m", "min_true_height_m"):
            if row.get(field) is None or not np.isfinite(row[field]):
                raise ValueError("Completed outcome has nonfinite physical diagnostics")
        for output_name, input_name in (("source_outward_entry_sample_s", "source_outward_entry_sample_s"),
            ("source_outward_entry_interpolated_s", "source_outward_entry_interpolated_s"),
            ("source_recovery_complete_s", "source_recovery_complete_s")):
            if not np.isclose(row[output_name], inputs[input_name][i], atol=1e-12, rtol=0):
                raise ValueError("Runtime record changed immutable full-source timing")
        if row["safe_arrival"]:
            returned = row.get("punch_return_time_s")
            source_end = float(inputs["source_recovery_complete_s"][i])
            if (returned is None or not np.isfinite(returned) or returned < source_end-1e-8
                    or returned-source_end >= .005+1e-8 or event < returned-1e-8):
                raise ValueError("Safe arrival precedes actual sampled reference completion")
    reference_snapshots = None
    for p, primitive in enumerate(primitives):
        expected_rows = np.flatnonzero(inputs["primitive_index"] == p)
        actual = [row for row in rows if row["primitive_id"] == primitive]
        if len(expected_rows) != 1000 or len(actual) != 1000:
            raise ValueError("Every unchanged primitive requires exactly1000 outcomes")
        expected_snapshots = inputs["snapshot_id"][expected_rows]
        if reference_snapshots is None:
            reference_snapshots = expected_snapshots
        np.testing.assert_array_equal(expected_snapshots, reference_snapshots,
            err_msg="All primitives must share the same1000 snapshot IDs/order")
        np.testing.assert_array_equal([row["episode_id"] for row in actual], ids[expected_rows],
            err_msg="Outcome order must match the original primitive/snapshot ordering")
        np.testing.assert_array_equal([row["snapshot_id"] for row in actual], expected_snapshots)
    if len(np.unique(reference_snapshots)) != 1000:
        raise ValueError("Matched snapshot set is not1000 unique states")
    return input_lookup


def summarize(runs, inputs_dir, output):
    inputs_dir = Path(inputs_dir).resolve(strict=True)
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT.resolve()):
        raise ValueError("Summary output must remain inside this diagnostic root")
    if output.exists() and (not output.is_dir() or any(
            p.name not in {"summary.json", "media_selection.json", "per_primitive.csv"}
            for p in output.iterdir())):
        raise FileExistsError("Output contains unrelated artifacts; use a fresh summary directory")
    manifest_path, episode_path = inputs_dir/"manifest.json", inputs_dir/"episodes.npz"
    manifest = json.loads(manifest_path.read_text())
    input_digest = digest(episode_path)
    if input_digest != manifest["episodes_sha256"]:
        raise ValueError("Frozen input archive checksum changed")
    with np.load(episode_path, allow_pickle=False) as source:
        inputs = {name: source[name].copy() for name in source.files}
    rows, provenance = [], []
    for run in runs:
        run = Path(run).resolve(strict=True)
        summary_path, invocation_path, rows_path = run/"summary.json", run/"invocation.json", run/"episodes.jsonl"
        summary, invocation = json.loads(summary_path.read_text()), json.loads(invocation_path.read_text())
        batch = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
        preserved = summary.get("unchanged_policy", {})
        if (summary.get("completed") != len(batch) or summary.get("learning") is not False
                or summary.get("source_clock_rate") != 1.
                or preserved.get("passed") is not True or not preserved.get("original")
                or preserved.get("original") != preserved.get("final")):
            raise ValueError(f"Run is incomplete or frozen policy preservation failed: {run}")
        if (invocation.get("inputs_sha256") != input_digest
                or invocation.get("manifest_sha256") != digest(manifest_path)
                or invocation.get("media_selection") is not None
                or invocation.get("learning") is not False):
            raise ValueError("Require original evaluation runs with exact matching inputs, not media replays")
        rows.extend(batch)
        provenance.append(dict(path=str(run), job_id=invocation["job_id"], completed=len(batch),
            unchanged_policy=preserved,
            files={p.name: digest(p) for p in (summary_path, invocation_path, rows_path)}))
    lookup = validate_rows(rows, inputs, manifest)
    policy_hashes = {run["unchanged_policy"]["checkpoint_sha256"] for run in provenance}
    if policy_hashes != {"116f37b6f99c9cd889def4959a5f93c96a521d83e041ad3b415bd9e53275b1cb"}:
        raise ValueError("All evaluations must use the user-selected H1 final checkpoint")
    primitives = manifest["primitive_ids"]
    summaries = {pid: describe([r for r in rows if r["primitive_id"] == pid], inputs, lookup) for pid in primitives}
    selected = select_representatives(rows, primitives)
    report = dict(schema="task10d_task08_8000_summary_v1", completed=True,
        source_clock_rate=1., camera_pitch_up_deg=25., checkpoint_sha256=next(iter(policy_hashes)),
        validation=dict(exact8000_unique_episode_ids=True, same1000_unique_snapshot_ids_and_order=True,
            all_completed=True, no_fatal_or_nonfinite=True, frozen_policy_preserved=True,
            exact_input_hashes=True, no_simulation_or_learning=True),
        overall=describe(rows, inputs, lookup), per_primitive=summaries,
        timing_semantics="Entrance times use the full unchanged source at the frozen launch target; runtime timing is the first terminal event. Safe-arrival-after-return is conditional on successful episodes.",
        scope="Same1000 TRAIN design snapshots for each of8 primitives; diagnostic, not untouched-test performance or full-bank coverage",
        runs=provenance, input_files={str(p): digest(p) for p in (manifest_path, episode_path)},
        source_sha256=digest(Path(__file__)), media_selection=dict(selected_total=selected["selected_total"],
            shortages=selected["shortages"], outcome_selected_illustrations=True))
    csv_rows = []
    for pid in primitives:
        item = summaries[pid]
        row = dict(primitive_id=pid, episodes=item["episodes"], safe_arrivals=item["safe_arrivals"],
            safe_arrival_rate=item["safe_arrival_rate"], initial_clean_both_count=item["initial_clean_both_count"],
            physical_hit_count=item["physical_hit_count"])
        row.update({"outcome_"+key: value for key, value in item["dispositions"].items()})
        for delay, value in item["by_delay_ms"].items():
            row[f"delay_{delay}ms_episodes"] = value["episodes"]
            row[f"delay_{delay}ms_safe_arrivals"] = value["safe_arrivals"]
        for name, values in item["timing"].items():
            row.update({name+"_"+field: values[field] for field in ("min", "median", "max")})
        csv_rows.append(row)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(csv_rows)
    artifacts = {"summary.json": json.dumps(report, indent=2, allow_nan=False)+"\n",
                 "media_selection.json": json.dumps(selected, indent=2, allow_nan=False)+"\n",
                 "per_primitive.csv": stream.getvalue()}
    for name, content in artifacts.items():
        verify_existing(output/name, content)
    output.mkdir(parents=True, exist_ok=True)
    for name, content in artifacts.items():
        path = output/name
        try:
            with path.open("x", newline="") as destination:
                destination.write(content)
        except FileExistsError:
            verify_existing(path, content)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path,
        default=[ROOT/"eval_nominal01", ROOT/"eval_variants01"])
    parser.add_argument("--inputs", type=Path, default=ROOT/"inputs01")
    parser.add_argument("--output", type=Path, default=ROOT/"analysis01")
    args = parser.parse_args()
    result = summarize(args.runs, args.inputs, args.output)
    print(json.dumps(dict(episodes=result["overall"]["episodes"],
        safe_arrivals=result["overall"]["safe_arrivals"],
        dispositions=result["overall"]["dispositions"], media_selection=result["media_selection"]), indent=2))


if __name__ == "__main__":
    main()
