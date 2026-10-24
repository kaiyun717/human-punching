"""Exact 8 x 1000 Phase3 inputs and unmodified frozen-head source timing.

No simulator is advanced and no policy is loaded. The historical source-path
cache and identity-matched clean native measurements are reused. Episode rows
point to the existing full-state geometry file; qvel/action history is neither
zeroed nor duplicated for every primitive. Interpolation is only a diagnostic
root inside an observed 5 ms entry bracket, never a clock adjustment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq

from .g1 import PROJECT, RESET_KEYS
from .phase1 import dump
from .phase2 import load_library
from .placement_phase3 import INDEX, INDEX_SHA256, PRIMITIVE_IDS, _rotate_xy
from .replay import Placement

PHASE3 = PROJECT / "outputs/task10d/phase3"
DIAGNOSTIC = PROJECT / "outputs/task10d/task08_diagnostic_20261024"
PATH_CACHE_SHA256 = "b345425f9ef6bb81e9135fdcb5353240ff72e96dc0f703f0fc536de388748a3f"
MANIFEST_SHA256 = "229398131c548bfbf5a97c54ab8df6cbb1c0d116c891e87d63cd23d111b1ba88"
FAMILIES = ("jab", "cross", "left_hook", "right_hook")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Input identity changed: {path}: {actual} != {expected}")
    return actual


def first_entry_timing(primitive, times, gloves, outward, head, placement,
                       *, interpolate=True):
    """Reuse cached samples; production FK only refines the first active bracket."""
    delta = placement.yaw - primitive.placement.yaw
    shift = np.asarray(placement.xy) - _rotate_xy(primitive.placement.xy, delta)
    query = np.asarray(head, float).copy()
    query[:2] = _rotate_xy(query[:2]-shift, -delta)
    distances = np.linalg.norm(gloves-query, axis=-1)
    active = ("left", "right").index(primitive.active_hand)
    hits = np.flatnonzero(outward & (distances[:, active] <= .14))
    if not len(hits) or hits[0] == 0:
        raise ValueError("Every selected input needs a future active-outward entry")
    first = int(hits[0])
    low, high = float(times[first-1]), float(times[first])
    if not distances[first-1, active] > .14:
        raise ValueError("The preceding source-grid sample must be outside the encounter")
    crossing = np.nan
    if interpolate:
        def signed_distance(time):
            sample = primitive.reference.sample(time, placement)
            return float(np.linalg.norm(sample.glove_pos[active]-head)-.14)
        crossing = float(brentq(signed_distance, low, high, xtol=1e-12))
        if not low <= crossing <= high:
            raise ValueError("Refined entry escaped the observed source-grid bracket")
    any_hits = np.flatnonzero((distances <= .14).any(-1))
    any_first = int(any_hits[0])
    inactive_hits = np.flatnonzero(distances[:,1-active] <= .14)
    earlier_inactive = inactive_hits[inactive_hits < first]
    return dict(source_outward_entry_sample_s=high,
        source_previous_sample_s=low, source_outward_entry_interpolated_s=crossing,
        source_outward_min_distance_m=float(distances[outward,active].min()),
        source_first_any_glove_entry_sample_s=float(times[any_first]),
        source_first_any_hand_index=int(np.argmin(distances[any_first])),
        source_inactive_first_entry_sample_s=np.nan if not len(inactive_hits) else float(times[inactive_hits[0]]),
        earlier_inactive_entry=False if not len(earlier_inactive) else True,
        initial_glove_distance_m=distances[0])


def verify_native_row(row, *, sid, primitive_id, rank, xy, yaw):
    """Check the original exact-placement clean measurement, not only its flag."""
    if (row["snapshot_id"], row["primitive_id"], row["pitch_up_deg"], row["rank"]) != (sid,primitive_id,25,rank):
        raise ValueError("Native evidence does not identify this episode placement")
    np.testing.assert_allclose(row["placement"]["xy"],xy,atol=1e-12,rtol=0)
    if abs(np.arctan2(np.sin(row["placement"]["yaw"]-yaw),np.cos(row["placement"]["yaw"]-yaw))) > 1e-12:
        raise ValueError("Native evidence yaw differs from selected placement")
    if not row["atomic_preserved"] or row["gross_intersection"] or row["threshold_m"] != -.03:
        raise ValueError("Selected native scene failed atomic or gross-intersection checks")
    metrics = row["metrics"]
    if metrics["capture_timestamps"] != 0 or metrics["delivered_capture_timestamps"] != 0 or metrics["padding"] is not True:
        raise ValueError("Initial observation timestamps/padding are not truthful")
    if not row["clean_both_represented"] or not row["rgb_both_represented"]:
        raise ValueError("Both hands must have confirmed clean C and ID information")
    for key in ("glove_valid_source_pixels","glove_rgb_valid_source_pixels",
                "glove_clean_min_cells","glove_acquired_min_cells"):
        values = np.asarray(metrics[key])
        if values.shape != (2,) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"Missing positive per-hand native evidence: {key}")
    return metrics


def build(output: Path, *, interpolate=True):
    output = output.resolve()
    if not output.is_relative_to(DIAGNOSTIC.resolve()):
        raise ValueError("Diagnostic inputs must stay inside their authorized output root")
    if output.exists():
        raise FileExistsError("Use a new input preparation directory")
    analysis_file = PHASE3/"analysis01/summary.json"
    analysis = json.loads(analysis_file.read_text())
    search_file = PHASE3/"search01/summary.json"
    require_hash(search_file,analysis["search_summary_sha256"])
    search = json.loads(search_file.read_text())
    geometry_path = PHASE3/"snapshots/geometry.npz"
    subset_path = PHASE3/"snapshots/training_subset.json"
    selected_path = PHASE3/"analysis01/selected_placements.npz"
    cache_path = PHASE3/"search01/cached_paths.npz"
    expected = {INDEX:INDEX_SHA256,INDEX.parent/"manifest.json":MANIFEST_SHA256,
        geometry_path:search["geometry_sha256"],subset_path:search["subset_sha256"],
        selected_path:analysis["selected_placements_sha256"],cache_path:PATH_CACHE_SHA256}
    for record in analysis["input_files"]:
        path = Path(record["path"])
        # Preserve the selected candidate/native inputs; the unrelated nominal
        # baseline and other pitches are not separate evidence for these rows.
        if path.parent.name == "search01" or path.parent.name == "shortlist03":
            expected[path] = record["sha256"]
            if "invocation_sha256" in record:
                expected[path.parent/"invocation.json"] = record["invocation_sha256"]
    for path,digest in expected.items():
        require_hash(path,digest)
    selected = dict(np.load(selected_path,allow_pickle=False))
    subset = json.loads(subset_path.read_text())
    ids = np.asarray(subset["snapshot_ids"],np.int64)
    if ids.shape != (1000,) or len(np.unique(ids)) != 1000:
        raise ValueError("Exactly the existing 1000 distinct TRAIN snapshot IDs are required")
    np.testing.assert_array_equal(selected["snapshot_id"],ids)
    np.testing.assert_array_equal(selected["primitive_id"],PRIMITIVE_IDS)
    p = int(np.flatnonzero(selected["pitch_up_deg"] == 25)[0])
    if not selected["geometric"][p].all() or not (selected["native_layer_status"][p] == 1).all() or not (selected["selected_class"][p] == 4).all():
        raise ValueError("All 8000 selected placements must already be geometrically and clean-C verified")
    geometry = dict(np.load(geometry_path,allow_pickle=False))
    lookup = {int(sid):i for i,sid in enumerate(geometry["snapshot_id"])}
    geometry_rows = np.array([lookup[int(sid)] for sid in ids],np.int64)
    np.testing.assert_array_equal(geometry_rows,subset["geometry_row_indices"])
    if not np.all(geometry["split_id"][geometry_rows] == 0):
        raise ValueError("The matched prior TRAIN subset must not be resplit")
    native_files = [Path(path) for path in analysis["native_measurement_paths"]]
    needed = {int(file):set(map(int,selected["selected_measurement_line"][p][selected["selected_measurement_file"][p]==file]))
              for file in np.unique(selected["selected_measurement_file"][p])}
    native = {}
    for file,lines in needed.items():
        if file < 0 or min(lines) < 1:
            raise ValueError("Selected native evidence is missing")
        native_path = native_files[file]
        if native_path not in expected:
            raise ValueError("Selected native measurement file lacks a verified input identity")
        invocation = json.loads((native_path.parent/"invocation.json").read_text())
        if invocation["geometry_sha256"] != search["geometry_sha256"] or invocation["dynamic"]:
            raise ValueError("Native measurements do not match the static snapshot geometry")
        with native_path.open() as stream:
            for line_number,line in enumerate(stream,1):
                if line_number in lines:
                    native[file,line_number] = json.loads(line)
    primitives = load_library(INDEX,list(PRIMITIVE_IDS))
    cache = np.load(cache_path,allow_pickle=False)
    arrays: dict[str,list] = {}
    summaries = []
    for j,primitive in enumerate(primitives):
        name = primitive.primitive_id
        times = cache[name+"__time_s"]
        gloves = cache[name+"__gloves"]
        outward = cache[name+"__outward"]
        if not np.array_equal(outward,times<=primitive.record["annotations"]["withdrawal_start_s"]+1e-12):
            raise ValueError("Cached outward window differs from accepted annotations")
        for i in (0,int(np.flatnonzero(outward)[-1]),len(times)-1):
            np.testing.assert_array_equal(gloves[i],primitive.reference.sample(float(times[i]),primitive.placement).glove_pos)
        with np.load(PHASE3/f"search01/{name}.npz",allow_pickle=False) as source:
            candidate = {key:source[key] for key in ("snapshot_id","top_available","top_xy",
                "top_yaw_rad","top_first_entry_sample_s","top_center_margin")}
            np.testing.assert_array_equal(candidate["snapshot_id"],ids)
            sampled_times = []
            refined_times = []
            for s,sid in enumerate(ids):
                rank = int(selected["selected_rank"][p,s,j])
                xy = selected["selected_xy"][p,s,j]
                yaw = float(selected["selected_yaw_rad"][p,s,j])
                if not candidate["top_available"][p,s,rank]:
                    raise ValueError("Chosen placement is no longer a retained geometric candidate")
                np.testing.assert_array_equal(xy,candidate["top_xy"][p,s,rank])
                if yaw != candidate["top_yaw_rad"][p,s,rank]:
                    raise ValueError("Selected placement differs from original search")
                file = int(selected["selected_measurement_file"][p,s,j])
                line = int(selected["selected_measurement_line"][p,s,j])
                measured = native[file,line]
                metrics = verify_native_row(measured,sid=int(sid),primitive_id=name,rank=rank,xy=xy,yaw=yaw)
                head = geometry["head_world"][geometry_rows[s]]
                timing = first_entry_timing(primitive,times,gloves,outward,head,Placement(tuple(xy),yaw),interpolate=interpolate)
                if timing["source_outward_entry_sample_s"] != candidate["top_first_entry_sample_s"][p,s,rank]:
                    raise ValueError("Frozen-head sampled entry changed from original placement evidence")
                if not (timing["initial_glove_distance_m"] >= .16).all():
                    raise ValueError("Selected initial glove clearance is unsafe")
                row = dict(episode_id=j*1000+s,snapshot_id=int(sid),snapshot_index=s,
                    geometry_row_index=int(geometry_rows[s]),primitive_index=j,family_index=FAMILIES.index(primitive.family),
                    active_hand_index=("left","right").index(primitive.active_hand),placement_xy=xy,
                    placement_yaw_rad=yaw,frozen_head_world=head,source_recovery_complete_s=float(primitive.reference.end_time_s),
                    source_outward_end_s=float(primitive.record["annotations"]["withdrawal_start_s"]),
                    source_clock_rate=1.,selected_geometric_rank=rank,initial_frustum=True,initial_clean_id=True,
                    initial_clean_c=True,initial_native_gross_pass=True,
                    initial_center_margin=candidate["top_center_margin"][p,s,rank],
                    native_measurement_file_index=file,native_measurement_line=line,
                    initial_native_convex_distance_capped_m=float(measured["minimum_native_convex_distance_capped_m"]),
                    **timing,**{key:np.asarray(metrics[key],np.int64) for key in (
                        "glove_valid_source_pixels","glove_rgb_valid_source_pixels","glove_clean_min_cells","glove_acquired_min_cells")})
                for key,value in row.items():
                    arrays.setdefault(key,[]).append(value)
                sampled_times.append(timing["source_outward_entry_sample_s"])
                refined_times.append(timing["source_outward_entry_interpolated_s"])
            summaries.append(dict(primitive_id=name,family=primitive.family,episodes=1000,
                source_outward_first_sample_s_min_median_max=np.quantile(sampled_times,[0,.5,1]),
                source_outward_first_interpolated_s_min_median_max=None if not interpolate else np.quantile(refined_times,[0,.5,1]),
                source_recovery_complete_s=primitive.reference.end_time_s,
                source_outward_end_s=primitive.record["annotations"]["withdrawal_start_s"],
                reference_npz=str(primitive.reference.path),reference_sha256=primitive.reference.sha256,
                reference_json_sha256=primitive.metadata_sha256))
            print(f"PREPARED {name} sample range/median {summaries[-1]['source_outward_first_sample_s_min_median_max']}",flush=True)
    payload = {key:np.asarray(value) for key,value in arrays.items()}
    validate_episode_matrix(payload,ids)
    output.mkdir(parents=True,exist_ok=False)
    episodes = output/"episodes.npz"
    np.savez_compressed(episodes,**payload)
    snapshot_hashes = {key:hashlib.sha256(np.ascontiguousarray(geometry[key][geometry_rows]).tobytes()).hexdigest()
        for key in (*RESET_KEYS,"world_from_collection")}
    manifest = dict(schema="task10d_task08_diagnostic_inputs_v1",created_utc=datetime.now(timezone.utc).isoformat(),
        episode_count=8000,snapshot_count=1000,primitive_ids=list(PRIMITIVE_IDS),families=list(FAMILIES),
        episode_order="primitive-major: episode_id = primitive_index*1000 + snapshot_index; original subset order retained",
        camera_pitch_up_deg=25,source_clock_rate=1.,requested_ttc_s=None,
        source_timing="5ms sampled active-outward entry bracket; optional production-FK brentq root inside that same bracket is diagnostic only",
        interpolation_enabled=interpolate,encounter_m=.14,initial_safe_clearance_m=.16,
        scope="Matched prior TRAIN design subset; eight unchanged primitives, no 3C full-bank coverage or learning",
        initial_observation="Existing uncorrupted actual native scene evidence; ordinary runtime corruption/delay is not used to admit episodes",
        reset_contract="Use geometry_row_index into geometry.npz; restore exact qpos/qvel and all three action histories through existing derived reset; preserve physical velocities, no settling step",
        reset_array_keys=list(RESET_KEYS),selected_snapshot_reset_array_sha256=snapshot_hashes,
        geometry_path=str(geometry_path),training_subset_path=str(subset_path),path_cache_path=str(cache_path),
        native_measurement_files=[str(path) for path in native_files],
        unique_snapshot_ids=ids,independence_note="Distinct stored IDs are not asserted to be independent trajectories; original split/provenance retained",
        primitives=summaries,episodes_sha256=sha256(episodes),
        earlier_inactive_before_active_count=int(payload["earlier_inactive_entry"].sum()),
        source_path=str(Path(__file__).resolve()),source_sha256=sha256(Path(__file__)),
        input_sha256={str(path):digest for path,digest in expected.items()},
        analysis_summary_sha256=sha256(analysis_file),command_argv=sys.argv[1:])
    dump(output/"manifest.json",manifest)
    return manifest


def validate_episode_matrix(arrays, snapshot_ids):
    """No dropped/repeated/swapped inputs in the required Cartesian product."""
    np.testing.assert_array_equal(arrays["episode_id"],np.arange(8000))
    np.testing.assert_array_equal(arrays["snapshot_id"],np.tile(snapshot_ids,8))
    np.testing.assert_array_equal(arrays["primitive_index"],np.repeat(np.arange(8),1000))
    np.testing.assert_array_equal(arrays["source_clock_rate"],1.)
    if np.any(arrays["source_outward_entry_sample_s"] > arrays["source_outward_end_s"]+1e-12):
        raise ValueError("Recovery-only entry must not count as outward coverage")
    if not np.all(arrays["source_previous_sample_s"] < arrays["source_outward_entry_sample_s"]):
        raise ValueError("Every first entry needs its preceding sampled bracket")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DIAGNOSTIC/"inputs01")
    parser.add_argument("--no-interpolation",action="store_true")
    args=parser.parse_args()
    build(args.output,interpolate=not args.no_interpolation)


if __name__ == "__main__":
    main()
