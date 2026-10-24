#!/usr/bin/env python3
"""Phase 3B-II common-initial-pelvis sample-ball reach maps; CPU FK only."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.spatial.transform import Rotation

from human_punching.task10d.common_start_phase3 import (
    accepted_to_common, common_start_transform, overlap_counts, sample_ball_membership,
)
from human_punching.task10d.placement_phase3 import cache_library, INDEX, DT, ENCOUNTER
from human_punching.task10d.g1 import PROJECT
from human_punching.task10c.robot import FOOT_BODY_NAMES

LABELS = ["Nominal jab", "Nominal cross", "Nominal left hook", "Nominal right hook",
          "Stepping jab", "Compact cross", "Tight left hook", "Tight right hook"]
BLUE, ORANGE, GREEN = "#1764a0", "#d55e00", "#009e73"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(fig, output: Path, name: str) -> list[str]:
    result = []
    for extension in ("png", "pdf"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        result.append(path.name)
    plt.close(fig)
    return result


def axes_style(ax, extent):
    ax.set_aspect("equal")
    ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
    ax.grid(alpha=.13)
    ax.plot(0, 0, marker="+", color="black", ms=7, mew=1.3)


def primitive_figure(output, x, y, z, masks, paths, trajectory):
    extent = [x[0], x[-1], y[0], y[-1]]
    fig, axes = plt.subplots(2, 4, figsize=(14.3, 8.8), sharex=True, sharey=True)
    fig.subplots_adjust(top=.81, bottom=.19, left=.07, right=.98, wspace=.12, hspace=.24)
    for j, (ax, path) in enumerate(zip(axes.flat, paths, strict=True)):
        # Each fixed-height mask is an exact membership query of the sample balls
        # at these grid centers; contour drawing does not fill a convex hull.
        ax.contourf(x, y, masks[1, j], levels=[.5, 1.5], colors=["#c6dbef"], alpha=.9)
        for height, color, linestyle, width in ((0, ORANGE, "--", 1.),
                (1, BLUE, "-", 1.3), (2, GREEN, ":", 1.4)):
            if masks[height, j].any():
                ax.contour(x, y, masks[height, j].astype(float), levels=[.5],
                           colors=[color], linestyles=[linestyle], linewidths=[width])
        row = trajectory[path.primitive.primitive_id]
        active = row["gloves"][path.outward, path.active_index]
        ax.plot(active[:, 0], active[:, 1], color="#242424", lw=1.1)
        ax.plot(active[0, 0], active[0, 1], "o", color="#242424", ms=3)
        ax.plot(active[-1, 0], active[-1, 1], "D", color="#242424", ms=3)
        for foot in range(2):
            ax.plot(row["feet"][:, foot, 0], row["feet"][:, foot, 1], color="#888888", lw=.9, alpha=.9)
        axes_style(ax, extent)
        ax.set_title(LABELS[j], fontsize=11)
        if j >= 4:
            ax.set_xlabel("Forward from initial H2 pelvis (m)")
        if j % 4 == 0:
            ax.set_ylabel("Left from initial H2 pelvis (m)")
    fig.suptitle("Eight accepted motions in one initial pelvis frame", y=.98, fontsize=18, fontweight="bold")
    fig.text(.53, .935, "One constant whole-motion transform per primitive: actual initial pelvis XY = (0, 0), heading = 0°",
             ha="center", fontsize=11)
    fig.legend(handles=[Patch(facecolor="#c6dbef", label=f"Head-center reach at nominal Z = {z[1]:.5f} m"),
        Line2D([0], [0], color=ORANGE, ls="--", label=f"Bank minimum Z = {z[0]:.5f} m"),
        Line2D([0], [0], color=GREEN, ls=":", label=f"Bank maximum Z = {z[2]:.5f} m")],
        loc="upper center", bbox_to_anchor=(.53, .908), ncol=3, frameon=False, fontsize=10)
    fig.text(.07, .11,
        "Black curve: XY projection of the active glove during the accepted preparation/outward interval (dot=start; diamond=end).\n"
        "Gray curves: complete-cycle foot tracks. Cross: initial pelvis XY. Height, articulation, stepping and recovery remain unchanged.",
        fontsize=9, color="#444444", linespacing=1.5)
    fig.text(.07, .035,
        "Shaded/outlined regions use the union of radius-0.14 m balls around production 5 ms samples, evaluated on a 5 mm XY grid.\n"
        "They are head-center reach regions for this eight-motion library, without camera/clearance/occlusion admission; not H2's full workspace.",
        fontsize=9, color="#444444", linespacing=1.5)
    return save(fig, output, "primitive_reach_common_frame")


def overlap_figure(output, x, y, z, primitive_count, family_count, union_area):
    extent = [x[0], x[-1], y[0], y[-1]]
    fig, axes = plt.subplots(3, 3, figsize=(12.6, 12.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.08, right=.96, top=.89, bottom=.19, wspace=.12, hspace=.21)
    primitive_map = ListedColormap(["white", *plt.get_cmap("viridis")(np.linspace(.12, .94, 8))])
    family_map = ListedColormap(["white", *plt.get_cmap("plasma")(np.linspace(.2, .85, 4))])
    count_plots = {}
    for h in range(3):
        for col, values in enumerate((primitive_count[h] > 0, primitive_count[h], family_count[h])):
            ax = axes[h, col]
            if col == 0:
                ax.contourf(x, y, values, levels=[.5, 1.5], colors=["#9ecae1"])
                title = f"Union · grid area {union_area[h]:.3f} m²"
            else:
                maximum, cmap = (8, primitive_map) if col == 1 else (4, family_map)
                plot = ax.pcolormesh(x, y, values, cmap=cmap,
                    norm=BoundaryNorm(np.arange(-.5, maximum + 1.5), cmap.N), shading="nearest")
                title = "Number of available primitives" if col == 1 else "Number of available families"
                if h == 2:
                    count_plots[col] = (plot, maximum)
            axes_style(ax, extent)
            ax.set_title(title, fontsize=10)
            if col == 0:
                role = ("Bank minimum", "Nominal", "Bank maximum")[h]
                ax.set_ylabel(f"{role} Z = {z[h]:.5f} m\nLeft (m)")
            if h == 2:
                ax.set_xlabel("Forward (m)")
    # Separate colorbar axes preserve identical physical panel dimensions and
    # aligned rows, in addition to the shared coordinate limits.
    fig.canvas.draw()
    for col, (plot, maximum) in count_plots.items():
        position = axes[2, col].get_position()
        cax = fig.add_axes([position.x0, .115, position.width, .014])
        bar = fig.colorbar(plot, cax=cax, orientation="horizontal", ticks=np.arange(maximum + 1))
        bar.set_label("Count (zero = outside the sampled reach union)", fontsize=8)
    fig.suptitle("Library reach union and overlap at actual head heights", y=.976, fontsize=17, fontweight="bold")
    fig.text(.52, .935,
        "Same initial H2 pelvis XY and heading for all motions · each jab/cross/hook family is counted once across its two variants",
        ha="center", fontsize=10)
    fig.text(.08, .055,
        "Reach membership is distance ≤0.14 m to at least one retained active-outward 5 ms glove center.\n"
        "All masks preserve disconnected regions and holes; no convex hull, enclosing sphere or segment-tube fill is used.",
        fontsize=9, color="#444444", linespacing=1.5)
    fig.text(.08, .018,
        "Grid areas approximate these fixed-height library regions; they are not state-bank coverage rates or the humanoid's full reachable workspace.",
        fontsize=9, color="#444444")
    return save(fig, output, "union_and_overlap_common_frame")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/task10d/phase3/3b_ii/common_frame_maps01")
    parser.add_argument("--geometry", type=Path, default=PROJECT / "outputs/task10d/phase3/snapshots/geometry.npz")
    parser.add_argument("--source-cache", type=Path, default=PROJECT / "outputs/task10d/phase3/search01/cached_paths.npz")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((PROJECT / "outputs/task10d/phase3/3b_ii").resolve()) or output.exists():
        raise ValueError("Use a new output directory beneath Phase3/3b_ii")
    geometry_meta = json.loads(args.geometry.with_suffix(".json").read_text())
    if sha256(args.geometry) != geometry_meta["payload_sha256"]:
        raise ValueError("Prepared snapshot geometry changed")
    with np.load(args.geometry, allow_pickle=False) as source:
        head, ids = source["head_world"], source["snapshot_id"]
    nominal = int(np.flatnonzero(ids == 53874)[0])
    z = np.array([head[:, 2].min(), head[nominal, 2], head[:, 2].max()])
    paths = cache_library()
    trajectory, details, payload = {}, {}, {}
    with np.load(args.source_cache, allow_pickle=False) as cached:
        for path in paths:
            pid = path.primitive.primitive_id
            for key, expected in (("time_s", path.time_s), ("gloves", path.gloves), ("outward", path.outward)):
                np.testing.assert_array_equal(cached[f"{pid}__{key}"], expected)
            placement = common_start_transform(path)
            gloves = accepted_to_common(path, path.gloves)
            samples = [path.primitive.reference.sample(float(t), path.primitive.placement) for t in path.time_s]
            body = np.stack([sample.body_pos for sample in samples])
            common = accepted_to_common(path, body)
            feet_ids = [path.primitive.reference.body_names.index(name) for name in FOOT_BODY_NAMES]
            pelvis, feet = common[:, 0], common[:, feet_ids]
            np.testing.assert_allclose(pelvis[0, :2], 0, atol=2e-14)
            np.testing.assert_array_equal(common[..., 2], body[..., 2])
            displacements = np.linalg.norm(common - common[0], axis=-1)
            source_displacements = np.linalg.norm(body - body[0], axis=-1)
            displacement_error = float(np.max(np.abs(displacements - source_displacements)))
            native_error, orientation_error = 0., 0.
            check_indices = np.unique([0, np.flatnonzero(path.outward)[-1], len(path.time_s)-1])
            for i in check_indices:
                native = path.primitive.reference.sample(float(path.time_s[i]), placement)
                native_error = max(native_error, float(np.max(np.abs(native.body_pos-common[i]))))
                expected_rotation = (placement.rotation * path.primitive.placement.rotation.inv()
                    * Rotation.from_quat(samples[i].body_quat, scalar_first=True)).as_matrix()
                orientation_error = max(orientation_error, float(np.max(np.abs(
                    Rotation.from_quat(native.body_quat, scalar_first=True).as_matrix()-expected_rotation))))
                np.testing.assert_array_equal(native.qpos, samples[i].qpos)
            if native_error > 2e-14 or displacement_error > 2e-14 or orientation_error > 2e-14:
                raise AssertionError("Constant whole-body registration failed")
            initial = path.primitive.reference.sample(0., placement)
            forward = Rotation.from_quat(initial.body_quat[0], scalar_first=True).as_matrix()[:, 0]
            heading = float(np.arctan2(forward[1], forward[0]))
            trajectory[pid] = dict(gloves=gloves, pelvis=pelvis, feet=feet)
            details[pid] = dict(family=path.primitive.family, active_hand=path.primitive.active_hand,
                reference_sha256=path.primitive.reference.sha256,
                accepted_initial_pelvis_world=path.initial_pelvis.tolist(),
                accepted_initial_heading_rad=path.initial_heading_rad, native_common_placement=asdict(placement),
                common_initial_pelvis=pelvis[0].tolist(), common_initial_heading_rad=heading,
                outward_interval_s=[float(path.time_s[path.outward][0]), float(path.time_s[path.outward][-1])],
                outward_sample_count=int(path.outward.sum()), active_outward_glove_z_range_m=[float(gloves[path.outward,path.active_index,2].min()), float(gloves[path.outward,path.active_index,2].max())],
                full_cycle_pelvis_net_displacement_xyz_m=(pelvis[-1]-pelvis[0]).tolist(),
                full_cycle_pelvis_path_length_m=float(np.linalg.norm(np.diff(pelvis, axis=0), axis=-1).sum()),
                full_cycle_foot_max_displacement_m_each=np.linalg.norm(feet-feet[0], axis=-1).max(0).tolist(),
                full_cycle_foot_net_displacement_xyz_m=(feet[-1]-feet[0]).tolist(),
                all_body_z_max_abs_error_m=0., all_body_displacement_norm_max_abs_error_m=displacement_error,
                native_full_body_sample_max_abs_error_m=native_error,
                native_orientation_matrix_max_abs_error=orientation_error,
                source_qpos_unchanged_at_native_checks=True)
            for key, value in dict(time_s=path.time_s, outward=path.outward, gloves=gloves, pelvis=pelvis, feet=feet).items():
                payload[f"{pid}__{key}"] = value
    active_points = np.concatenate([trajectory[p.primitive.primitive_id]["gloves"][p.outward,p.active_index] for p in paths])
    grid_step = .005
    low = np.minimum(active_points[:, :2].min(0)-ENCOUNTER-.03, [-.08, -.08])
    high = np.maximum(active_points[:, :2].max(0)+ENCOUNTER+.03, [.08, .08])
    x = np.arange(np.floor(low[0]/grid_step), np.ceil(high[0]/grid_step)+1) * grid_step
    y = np.arange(np.floor(low[1]/grid_step), np.ceil(high[1]/grid_step)+1) * grid_step
    xx, yy = np.meshgrid(x, y)
    masks = np.empty((len(z), len(paths), len(y), len(x)), dtype=bool)
    for h, height in enumerate(z):
        queries = np.stack([xx, yy, np.full_like(xx, height)], -1)
        for j, path in enumerate(paths):
            active = trajectory[path.primitive.primitive_id]["gloves"][path.outward, path.active_index]
            masks[h, j] = sample_ball_membership(active, queries)
    counts = [overlap_counts(mask, tuple(path.primitive.family for path in paths)) for mask in masks]
    primitive_count, family_count = np.stack([row[0] for row in counts]), np.stack([row[1] for row in counts])
    areas = (primitive_count > 0).sum((-2,-1)) * grid_step**2
    output.mkdir(parents=True)
    np.savez_compressed(output / "common_frame_geometry.npz", x_m=x, y_m=y, head_z_m=z,
        primitive_ids=np.asarray([p.primitive.primitive_id for p in paths]),
        primitive_masks=masks, primitive_count=primitive_count, family_count=family_count, **payload)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "axes.spines.top":False,
                         "axes.spines.right":False, "pdf.fonttype":42})
    figures = primitive_figure(output, x, y, z, masks, paths, trajectory)
    figures += overlap_figure(output, x, y, z, primitive_count, family_count, areas)
    summary = dict(schema="task10d_phase3b_ii_common_frame_sample_ball_maps_v1",
        scope="Unchanged eight-motion library-specific geometric reach in a common actual initial pelvis frame; no camera, scene-collision, avoidance or full-workspace claim",
        source_index_sha256=sha256(INDEX), source_production_path_cache_sha256=sha256(args.source_cache),
        source_geometry_sha256=sha256(args.geometry), script_sha256=sha256(Path(__file__)),
        initial_frame="Actual initial H2 pelvis XY=(0,0), projected heading=0; one constant world-up yaw/XY transform for each complete motion; world Z unchanged",
        reach="Union of closed radius-0.14m balls around active-hand source-grid points in the accepted preparation/outward interval. No interpolation tube, convex hull or enclosing volume fill.",
        source_sample_interval_s=DT, sphere_radius_m=ENCOUNTER, xy_grid_step_m=grid_step,
        raster_note="Mask membership is exact at XY grid centers for the retained samples; plotted contours/grid areas carry 5mm planar raster approximation. No continuous-time crossing claim.",
        head_z_roles=["actual eligible bank minimum", "nominal snapshot53874", "actual eligible bank maximum"],
        head_z_m=z.tolist(), union_grid_area_m2=areas.tolist(),
        maximum_primitive_overlap=primitive_count.max((-2,-1)).tolist(),
        maximum_family_overlap=family_count.max((-2,-1)).tolist(),
        primitive_area_m2={p.primitive.primitive_id:(masks[:,j].sum((-2,-1))*grid_step**2).tolist() for j,p in enumerate(paths)},
        primitive_count_area_m2=[(np.bincount(row.ravel(), minlength=9)*grid_step**2).tolist() for row in primitive_count],
        family_count_area_m2=[(np.bincount(row.ravel(), minlength=5)*grid_step**2).tolist() for row in family_count],
        zero_count_area_note="Zero-count area depends on the finite displayed rectangle; positive-count areas describe this library's sampled reach slices.",
        primitives=details, files={name:sha256(output/name) for name in ["common_frame_geometry.npz",*figures]},
        physics_steps=0, camera_rendering=False, full_bank_coverage=False)
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(output), head_z_m=z.tolist(), union_grid_area_m2=areas.tolist(),
                         max_native_error_m=max(row["native_full_body_sample_max_abs_error_m"] for row in details.values())), indent=2))


if __name__ == "__main__":
    main()
