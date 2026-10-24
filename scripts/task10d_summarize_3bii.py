#!/usr/bin/env python3
"""Join bounded common-pelvis geometry to static, identity-matched RGB-D captures.

Unrendered clean observations remain unknown. This reads existing evidence;
it does not run dynamics, render new scenes, or change the Phase 3C procedure.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
STAGES = ("reach_only", "reach_within_allowed_envelope",
          "reach_frustum_clearance_allowed", "reach_frustum_initial_scene_valid")
METRICS = ("glove_valid_source_pixels", "glove_rgb_valid_source_pixels",
           "glove_clean_cells", "glove_acquired_cells",
           "glove_clean_min_cells", "glove_acquired_min_cells")
LABELS = ("Jab", "Cross", "Left hook", "Right hook", "Stepping jab", "Compact cross",
          "Tight left hook", "Tight right hook")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def key(row):
    return row["common_pose_id"], row["primitive_id"]


def same_placement(actual, expected):
    np.testing.assert_allclose(actual["xy"], expected["xy"], atol=1e-12, rtol=0)
    # Compare the stored transform itself, not an alternative modulo-2pi record.
    np.testing.assert_allclose(actual["yaw"], expected["yaw"], atol=1e-12, rtol=0)


def observation(row):
    if row.get("atomic_preserved") is not True:
        raise ValueError("Native atomic reset preservation failed")
    metrics = row["metrics"]
    if (metrics["capture_timestamps"] != 0 or metrics["delivered_capture_timestamps"] != 0
            or metrics["padding"] is not True):
        raise ValueError("Initial acquisition timestamps/padding are not truthful")
    for name in METRICS:
        values = np.asarray(metrics[name])
        if (values.shape != (2,) or not np.isfinite(values).all()
                or np.any(values < 0) or np.any(values != np.floor(values))):
            raise ValueError(f"Invalid per-hand observation counts: {name}")
    clean_id = bool(np.all(np.asarray(metrics["glove_valid_source_pixels"]) > 0)
                    and np.all(np.asarray(metrics["glove_clean_min_cells"]) > 0))
    clean_rgb = bool(np.all(np.asarray(metrics["glove_rgb_valid_source_pixels"]) > 0)
                     and np.all(np.asarray(metrics["glove_acquired_min_cells"]) > 0))
    if clean_id != row["clean_both_represented"] or clean_rgb != row["rgb_both_represented"]:
        raise ValueError("Observation flags disagree with native source/minimum-cell evidence")
    if (type(row.get("gross_intersection")) is not bool or row.get("threshold_m") != -.03
            or row["gross_intersection"] != (row["minimum_native_convex_distance_capped_m"] < -.03)):
        raise ValueError("Native gross-intersection semantics changed")
    return clean_id, clean_rgb


def counts(rows, field):
    passed = [row for row in rows if row[field] is True]
    return dict(passed_primitives=len(passed), passed_families=len({r["family"] for r in passed}),
                failed_primitives=sum(row[field] is False for row in rows),
                unknown_primitives=sum(row[field] is None for row in rows))


def nominal_matrix(state, primitive_ids, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    poses = state["candidates"]
    values = np.array([[sum(bool(row[stage]) for stage in STAGES)
                        for row in pose["evaluations"]] for pose in poses])
    fig, ax = plt.subplots(figsize=(11.5, max(5., .29 * len(poses) + 2.2)))
    ax.imshow(values, vmin=0, vmax=4, aspect="auto",
              cmap=ListedColormap(["#eeeeee", "#f4d5ac", "#e5dcaa", "#b7d8e4", "#a6d5bd"]))
    for i, pose in enumerate(poses):
        for j, row in enumerate(pose["evaluations"]):
            stage = ("-", "R", "A", "F", "S")[values[i, j]]
            clean = row["rendered_clean_rgb_hsv_both"]
            ax.text(j, i, stage + " / C" + ("?" if clean is None else "+" if clean else "-"),
                    ha="center", va="center", fontsize=8)
    ax.set(xticks=range(len(primitive_ids)), xticklabels=LABELS,
           yticks=range(len(poses)), yticklabels=[p["common_pose_id"].split("_", 1)[1]
               + (" *" if p["common_pose_id"] == state["best_pose_id"] else "") for p in poses],
           xlabel="Unchanged primitive at the same actual initial pelvis XY/yaw",
           ylabel="Candidate common pose", title=f"Nominal snapshot {state['snapshot_id']}: bounded common-pose candidates")
    ax.tick_params(axis="both", length=0)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.text(.13, .035, "Stages: - no raw reach; R raw reach; A reach + allowed envelope; "
             "F + frustum/clearance; S + native gross check.\n"
             "C+ / C-: measured clean RGB-HSV both-hand representation; C?: unrendered unknown. "
             "* selected pose.\nEach cell reports separate geometry and observation evidence; "
             "C+ alone does not make a motion admissible.", fontsize=8)
    fig.subplots_adjust(left=.13, right=.99, top=.94, bottom=.14)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"nominal_common_pose_matrix.{suffix}", dpi=170, facecolor="white")
    plt.close(fig)


def flag(value):
    return "?" if value is None else "+" if value else "-"


def nominal_sheet(state, render_dir, output):
    pose = next(p for p in state["candidates"] if p["common_pose_id"] == state["best_pose_id"])
    sheet = Image.new("RGB", (1344, 1120), "#151b23")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), f"STATIC t=0 | {pose['common_pose_id']} | exact common actual pelvis XY/yaw; motion articulation differs", fill="white")
    xy, yaw = pose["actual_initial_pelvis_xy"], pose["actual_initial_pelvis_yaw_rad"]
    draw.text((12, 32), f"XY=({xy[0]:.6f}, {xy[1]:.6f}) m; yaw={np.rad2deg(yaw):.6f} deg | +25 deg camera | no dynamics", fill="white")
    inputs = []
    for i, row in enumerate(pose["evaluations"]):
        x, y = 672 * (i % 2), 64 + 264 * (i // 2)
        title = (f"{LABELS[i]} | raw R{flag(row['reach_only'])} "
                 f"allowed A{flag(row['reach_within_allowed_envelope'])} "
                 f"scene S{flag(row['reach_frustum_initial_scene_valid'])} "
                 f"clean C{flag(row['rendered_clean_rgb_hsv_both'])}")
        draw.text((x + 8, y + 4), title, fill="white")
        draw.text((x + 8, y + 24), "External oblique                         Actual mounted RGB", fill="white")
        if row["rendered"]:
            path = render_dir / "initial_scenes" / f"{row['render_label']}.png"
            with Image.open(path) as panel:
                if panel.size != (2016, 848):
                    raise ValueError(f"Unexpected six-panel dimensions: {path}")
                sheet.paste(panel.crop((0, 48, 672, 424)).resize((336, 188)), (x, y + 44))
                sheet.paste(panel.crop((1344, 48, 2016, 424)).resize((336, 188)), (x + 336, y + 44))
            inputs.append(dict(path=str(path), sha256=sha256(path)))
        else:
            draw.text((x + 10, y + 100), "UNRENDERED: clean observation unknown", fill="#ffc879")
        if row["rendered"]:
            draw.text((x + 8, y + 240), "C source pixels L/R=" + str(row["metrics"]["glove_rgb_valid_source_pixels"])
                      + "; C union-min cells L/R=" + str(row["metrics"]["glove_acquired_min_cells"]), fill="white")
    sheet.save(output / "nominal_initial_contact_sheet.jpg", quality=94)
    return inputs


def summarize(placements, render_dir, output):
    placements, render_dir = placements.resolve(strict=True), render_dir.resolve(strict=True)
    if placements.is_dir():
        placements = placements / "common_placements.json"
    output = output.resolve()
    if not output.is_relative_to((PROJECT / "outputs/task10d/phase3/3b_ii").resolve()):
        raise ValueError("Use a new output directory within Phase3/3b_ii")
    if output.exists():
        raise FileExistsError(output)
    measurement_path, invocation_path = render_dir / "measurements.jsonl", render_dir / "invocation.json"
    completion_path = render_dir / "summary.json"
    original = json.loads(placements.read_text())
    invocation, completion = json.loads(invocation_path.read_text()), json.loads(completion_path.read_text())
    # Compute/login nodes expose the same project through distinct mount aliases.
    # Match the exact project-relative path and recorded content hash, not inode.
    request_path = (placements.parent / Path(invocation["requests"]).name).resolve(strict=True)
    expected_suffix = "/human-punching/" + str(request_path.relative_to(PROJECT))
    if not str(invocation["requests"]).endswith(expected_suffix):
        raise ValueError("Native request set must belong to the supplied placement evidence")
    request_record = json.loads(request_path.read_text())
    requests = request_record["requests"]
    if original["schema"] != "task10d_3b_ii_common_actual_pelvis_v1" or original["pitch_up_deg"] != 25:
        raise ValueError("Unexpected common-pelvis study or camera")
    if invocation["dynamic"] or invocation["requests_sha256"] != sha256(request_path):
        raise ValueError("Expected static rendering of these exact requests")
    geometry_hashes = [digest for path, digest in original["inputs"].items() if path.endswith("/geometry.npz")]
    if geometry_hashes != [invocation["geometry_sha256"]]:
        raise ValueError("Placement and rendering geometry identities differ")
    expected, poses, states = {}, {}, deepcopy(original["states"])
    primitive_ids = original["primitive_ids"]
    if len(primitive_ids) != 8 or len(set(primitive_ids)) != 8:
        raise ValueError("Expected all eight unchanged primitives")
    for state in states:
        for pose in state["candidates"]:
            if pose["common_pose_id"] in poses:
                raise ValueError("Duplicate common pose identity")
            poses[pose["common_pose_id"]] = (state, pose)
            if [r["primitive_id"] for r in pose["evaluations"]] != primitive_ids:
                raise ValueError("Primitive ordering/coverage differs across common poses")
            for row in pose["evaluations"]:
                for previous, following in zip(STAGES, STAGES[1:]):
                    if row[following] and not row[previous]:
                        raise ValueError("Geometric-stage nesting is inconsistent")
                row.update(rendered=False, rendered_clean_id_both=None, rendered_clean_rgb_hsv_both=None,
                           rendered_joint_clean_rgb_hsv=None, rendered_scene_valid=None)
    for request in requests:
        identity = key(request)
        if identity in expected:
            raise ValueError("Duplicate render request")
        state, pose = poses[request["common_pose_id"]]
        row = next(r for r in pose["evaluations"] if r["primitive_id"] == request["primitive_id"])
        if request["snapshot_id"] != state["snapshot_id"] or request["pitch_up_deg"] != 25:
            raise ValueError("Request snapshot/pitch differs from candidate")
        same_placement(request["placement"], row["placement"])
        np.testing.assert_array_equal(request["common_initial_pelvis_xy"], pose["actual_initial_pelvis_xy"])
        np.testing.assert_equal(request["common_initial_pelvis_yaw_rad"], pose["actual_initial_pelvis_yaw_rad"])
        np.testing.assert_equal(request["expected_initial_pelvis_z_m"], row["actual_initial_pelvis_z_m"])
        expected[identity] = request, row
    measurements = [json.loads(line) for line in measurement_path.read_text().splitlines() if line.strip()]
    if completion["completed"] != len(measurements) or completion["job_id"] != invocation["job_id"]:
        raise ValueError("Native completion record does not match measurements")
    seen, gross_disagreements = set(), []
    for actual in measurements:
        identity = key(actual)
        if identity not in expected or identity in seen:
            raise ValueError("Unexpected or duplicate native measurement")
        seen.add(identity)
        request, row = expected[identity]
        for field in ("snapshot_id", "pitch_up_deg", "primitive_id", "common_pose_id", "label", "kind"):
            if actual[field] != request[field]:
                raise ValueError(f"Native identity mismatch: {field}")
        same_placement(actual["placement"], request["placement"])
        native_pelvis = np.asarray(actual["actual_initial_pelvis_world"])
        desired = [*request["common_initial_pelvis_xy"], request["expected_initial_pelvis_z_m"]]
        np.testing.assert_allclose(native_pelvis, desired, atol=5e-7, rtol=0)
        clean_id, clean_rgb = observation(actual)
        gross = actual["gross_intersection"]
        if gross != row["native_initial_intersection"]["gross_intersection"]:
            gross_disagreements.append(list(identity))
        row.update(rendered=True, rendered_clean_id_both=clean_id, rendered_clean_rgb_hsv_both=clean_rgb,
                   clean_observation_status="passed" if clean_rgb else "failed",
                   rendered_scene_valid=bool(row["reach_frustum_clearance_allowed"] and not gross),
                   rendered_joint_clean_rgb_hsv=bool(row["reach_frustum_clearance_allowed"] and not gross and clean_rgb),
                   render_label=actual["label"], metrics=actual["metrics"],
                   actual_initial_pelvis_world=native_pelvis.tolist(), atomic_preserved=True,
                   rendered_native_intersection={name: actual[name] for name in
                       ("gross_intersection", "minimum_native_convex_distance_capped_m", "closest_pair")})
    selected = []
    for state in states:
        for pose in state["candidates"]:
            rows = pose["evaluations"]
            pose["postrender_counts"] = {stage: counts(rows, stage) for stage in (*STAGES,
                "rendered_scene_valid", "rendered_clean_rgb_hsv_both", "rendered_joint_clean_rgb_hsv")}
            if any((pose["common_pose_id"], row["primitive_id"]) in expected for row in rows):
                rendered = [row for row in rows if row["rendered"]]
                if rendered:
                    xy = np.array([row["actual_initial_pelvis_world"][:2] for row in rendered])
                    spread = float(np.ptp(xy, axis=0).max())
                    if spread > 1e-6:
                        raise ValueError("Common actual pelvis XY differs across primitive scenes")
                else:
                    spread = None
                pose["native_pelvis_audit"] = dict(rendered_count=len(rendered),
                    maximum_component_xy_spread_m=spread, each_matches_requested_xy_and_original_z=True if rendered else None,
                    native_yaw_check="runtime assertion at 5e-7 rad; actual yaw not exported separately")
                selected.append(dict(snapshot_id=state["snapshot_id"], common_pose_id=pose["common_pose_id"],
                    actual_initial_pelvis_xy=pose["actual_initial_pelvis_xy"],
                    actual_initial_pelvis_yaw_rad=pose["actual_initial_pelvis_yaw_rad"],
                    counts=pose["postrender_counts"], native_pelvis_audit=pose["native_pelvis_audit"], evaluations=rows))
    report = dict(schema="task10d_3b_ii_static_postrender_v1", camera_pitch_up_deg=25,
        native_job_id=invocation["job_id"], candidate_pose_count=len(poses), requested_scene_count=len(requests),
        render_selection=request_record.get("selection"),
        pre_render_request_metadata={name: value for name, value in request_record.items() if name != "requests"},
        measured_scene_count=len(measurements), missing_requests=[list(k) for k in expected if k not in seen],
        native_gross_check_disagreements=gross_disagreements, selected_poses=selected, states=states,
        scope="Static initial scenes only; no dynamics, avoidance success, switching, or Phase3C execution",
        clean_semantics="Uncorrupted RGB-HSV source pixels and actual unchanged 16x9 union minima must represent BOTH hands. Per-hand counts are acquisition diagnostics; ties may credit both hands. Unrendered is unknown.",
        source_sampling="Active hand at retained 5 ms production samples through withdrawal start; no recovery-only reach",
        placement_scope="Finite donor common-pose set within existing recipient numerical envelopes; not exhaustive common-pose impossibility",
        source_sha256=sha256(Path(__file__)),
        inputs=[dict(path=str(p), sha256=sha256(p)) for p in
                (placements, request_path, measurement_path, invocation_path, completion_path)])
    output.mkdir(parents=True, exist_ok=False)
    nominal = next(state for state in states if state["snapshot_id"] == 53874)
    nominal_matrix(nominal, primitive_ids, output)
    report["contact_sheet_sources"] = nominal_sheet(nominal, render_dir, output)
    write_json(output / "summary.json", report)
    lines = ["# Common actual-pelvis static scene review", "",
        f"Camera: +25 degrees upward. Native job {invocation['job_id']}; {len(measurements)}/{len(requests)} requested initial scenes measured. No dynamics or Phase 3C.", "",
        "Render selection: " + str(request_record.get("selection", "See exact input request record.")), "",
        "R = raw active-outward reach; A = reach within allowed envelope; F = A plus initial frustum/clearance; S = F plus native gross check. C = measured clean RGB-HSV representation of both hands; J = rendered scene validity plus C. Unrendered C/J remain unknown.", "",
        "| Snapshot / common pose | R primitives/families | A | F | S | C pass/fail/unknown | J pass/fail/unknown |", "|---|---:|---:|---:|---:|---:|---:|"]
    for pose in selected:
        c = pose["counts"]
        stage_cells = [f"{c[s]['passed_primitives']}/{c[s]['passed_families']}" for s in STAGES]
        native_cells = ["/".join(str(c[s][k]) for k in ("passed_primitives", "failed_primitives", "unknown_primitives"))
                        for s in ("rendered_clean_rgb_hsv_both", "rendered_joint_clean_rgb_hsv")]
        lines.append("| " + " | ".join([pose["common_pose_id"], *stage_cells, *native_cells]) + " |")
    for pose in selected:
        lines.extend(["", f"## {pose['common_pose_id']}", "",
            f"Actual initial pelvis XY={pose['actual_initial_pelvis_xy']}, yaw={pose['actual_initial_pelvis_yaw_rad']:.9f} rad. Original per-motion Z retained; XY maximum component spread={pose['native_pelvis_audit']['maximum_component_xy_spread_m']} m.", "",
            "| Primitive | R/A/F/S | C | J | Valid ID source L/R | Valid C source L/R | ID union-min L/R | C union-min L/R |", "|---|---|---|---|---|---|---|---|"])
        for row in pose["evaluations"]:
            values = [row["primitive_id"], "/".join(flag(row[s]) for s in STAGES),
                      flag(row["rendered_clean_rgb_hsv_both"]), flag(row["rendered_joint_clean_rgb_hsv"])]
            values.extend(str(row.get("metrics", {}).get(s, "unknown")) for s in
                ("glove_valid_source_pixels", "glove_rgb_valid_source_pixels", "glove_clean_min_cells", "glove_acquired_min_cells"))
            lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "[Nominal candidate matrix](nominal_common_pose_matrix.png) · [Nominal eight-scene contact sheet](nominal_initial_contact_sheet.jpg)", "",
        "Native intersection uses convex geometry and the declared -3 cm gross threshold; it does not establish collision freedom. Clean C is an acquisition diagnostic, not continuous visibility or delivered policy information. These finite candidate results do not establish impossibility outside the tested pose set.", "",
        f"Missing requests: {report['missing_requests']}; CPU/native gross-check disagreements: {gross_disagreements}."])
    (output / "README.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placements", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.placements, args.render_dir, args.output)
    print(json.dumps({key: result[key] for key in ("measured_scene_count", "missing_requests",
                                                  "native_gross_check_disagreements")}, indent=2))


if __name__ == "__main__":
    main()
