"""Render one nominal H2 jab adaptation against an unchanged real training head.

Stored poses are interpolated and displayed using native FK only. The head is
a non-solid visualization marker; no simulation, contact solve, or policy runs.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from human_punching.task10c.references import StoredReference
from human_punching.task10c.robot import GLOVE_RADIUS, HEAD_RADIUS, H2_XML, ROOT
from task10c_compare_references import add_shape


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def time_grid(end: float, dt: float) -> np.ndarray:
    times = np.arange(0.0, end + 1e-9, dt)
    return np.r_[times, end] if times[-1] < end - 1e-9 else times


class PoseView:
    def __init__(self, path: Path, target: np.ndarray, panel: int):
        self.path = path.resolve()
        self.reference = StoredReference.load(self.path)
        if self.reference.metadata["family"] != "jab":
            raise ValueError("This focused comparison accepts jab references only")
        spec = mujoco.MjSpec.from_file(str(H2_XML))
        spec.worldbody.add_geom(name="preview_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=[8, 8, .1], rgba=[.82, .84, .87, 1],
                                contype=0, conaffinity=0)
        spec.worldbody.add_light(pos=[2, -2, 4], dir=[-1, 1, -2], diffuse=[.8, .8, .8])
        self.model = spec.compile()
        self.model.vis.global_.offwidth = panel
        self.model.vis.global_.offheight = panel
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site("left_glove_center").id
        for gid in range(self.model.ngeom):
            name = self.model.geom(gid).name or ""
            if name.endswith("_collision") and "glove" not in name:
                self.model.geom_rgba[gid, 3] = 0.0
        self.target = target
        self.end = float(self.reference.time_s[-1])
        self.strike = float(self.reference.metadata["strike_time_s"])
        self.hold = float(self.reference.metadata.get("terminal_guard_hold_s", .4))

    def set_time(self, time_s: float) -> None:
        self.data.qpos[:] = self.reference.sample(time_s)["qpos"]
        mujoco.mj_kinematics(self.model, self.data)

    def measures(self) -> dict:
        times = time_grid(self.end, .005)
        poses = self.reference.sample(times)["qpos"]
        positions = []
        for qpos in poses:
            self.data.qpos[:] = qpos
            mujoco.mj_kinematics(self.model, self.data)
            positions.append(self.data.site_xpos[self.site].copy())
        positions = np.asarray(positions)
        distance = np.linalg.norm(positions - self.target, axis=1)
        entered = np.flatnonzero(distance <= GLOVE_RADIUS + HEAD_RADIUS)
        closest = int(distance.argmin())
        self.set_time(self.strike)
        return {
            "reference": str(self.path), "reference_sha256": sha256(self.path),
            "metadata_sha256": sha256(self.path.with_suffix(".json")),
            "nominal_strike_annotation_s": self.strike,
            "end_time_s": self.end, "terminal_hold_display_s": self.hold,
            "first_entry_s": None if not len(entered) else float(times[entered[0]]),
            "first_entry_previous_sample_s": None if not len(entered) or entered[0] == 0
                else float(times[entered[0] - 1]),
            "launch_overlap": bool(distance[0] <= GLOVE_RADIUS + HEAD_RADIUS),
            "closest_glove_center_time_s": float(times[closest]),
            "minimum_glove_center_distance_m": float(distance[closest]),
            "glove_center_at_nominal_strike_m": self.data.site_xpos[self.site].tolist(),
            "center_distance_at_nominal_strike_m": float(np.linalg.norm(
                self.data.site_xpos[self.site] - self.target)),
            "measurement": "Native site FK of interpolated stored qpos on a 5 ms grid; first center distance <=0.14 m. Not simulated first contact or continuous-time certification.",
        }


def label(value: float | None) -> str:
    return "none" if value is None else f"{value:.3f}s"


def annotated(images: list[Image.Image], title: str, lines: list[str]) -> Image.Image:
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height + 96), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill="black")
    for row, line in enumerate(lines):
        draw.text((10, 27 + 18 * row), line, fill="black")
    offset = 0
    for image in images:
        canvas.paste(image, (offset, 96))
        offset += image.width
    return canvas


def sheet(tiles: list[Image.Image], columns: int, path: Path) -> None:
    width, height = tiles[0].size
    canvas = Image.new("RGB", (columns * width, int(np.ceil(len(tiles) / columns)) * height), "white")
    for i, tile in enumerate(tiles):
        canvas.paste(tile, ((i % columns) * width, (i // columns) * height))
    canvas.save(path, quality=92)


def render(before: Path, after: Path, target_json: Path, output: Path) -> None:
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Run this renderer inside the authorized allocated GPU job")
    output = output.resolve()
    if not output.is_relative_to(ROOT.resolve()):
        raise ValueError("Comparison outputs must remain inside human-punching")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve existing comparison artifacts; choose an empty output directory")
    target_data = json.loads(target_json.read_text())
    target = np.asarray(target_data["target_world"], dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("target.json needs one finite target_world XYZ")
    if before.resolve() == after.resolve():
        raise ValueError("Before and after must be separate preserved reference files")
    panel, fps = 480, 50
    views = [PoseView(path, target, panel) for path in (before, after)]
    metrics = [view.measures() for view in views]
    cameras = []
    for azimuth in (135, 90):
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 3.6, azimuth, -10
        camera.lookat[:] = [.35, .05, 1.0]
        cameras.append(camera)
    options = mujoco.MjvOption()
    options.geomgroup[2] = 1
    options.geomgroup[3] = 1
    times = time_grid(max(view.end + view.hold for view in views), 1 / fps)
    dense_indices = set(range(0, len(times), 5)) | {len(times) - 1}
    selected_times = [0., .5 * views[1].strike, views[1].strike,
                      views[1].strike + .3, views[1].end, float(times[-1])]
    key_indices = {int(np.argmin(abs(times - t))) for t in selected_times}
    dense_tiles, key_tiles = [], []
    output.mkdir(parents=True, exist_ok=True)
    renderers = []
    try:
        renderers = [mujoco.Renderer(view.model, height=panel, width=panel) for view in views]
        with ExitStack() as stack:
            writers = [stack.enter_context(imageio.get_writer(output / name, fps=fps,
                       codec="libx264", quality=8, macro_block_size=2))
                       for name in ("before.mp4", "after.mp4", "comparison.mp4")]
            for frame_id, time_s in enumerate(times):
                images = []
                for view, renderer in zip(views, renderers):
                    view.set_time(float(time_s))
                    camera_images = []
                    for camera in cameras:
                        renderer.update_scene(view.data, camera, scene_option=options)
                        # Visual-only sphere: never added to the dynamics model.
                        add_shape(renderer.scene, mujoco.mjtGeom.mjGEOM_SPHERE,
                                  [HEAD_RADIUS] * 3, target, (1., .72, .05, .8))
                        camera_images.append(Image.fromarray(renderer.render().copy()))
                    images.append(camera_images)
                for i, title in enumerate(("BEFORE: original nominal jab", "AFTER: one adapted nominal jab")):
                    detail = metrics[i]
                    frame = annotated(images[i], f"{title} | t={time_s:.2f}s | oblique / side", [
                        f"Nominal strike annotation={detail['nominal_strike_annotation_s']:.3f}s; first 0.14m entry={label(detail['first_entry_s'])} (5ms FK)",
                        f"Closest center t={detail['closest_glove_center_time_s']:.3f}s, distance={detail['minimum_glove_center_distance_m']:.3f}m; TRAIN snapshot {target_data['snapshot_id']}",
                        "Kinematic poses only; fixed non-solid yellow head r=0.06m; red/blue gloves r=0.08m; no dynamics."])
                    writers[i].append_data(np.asarray(frame))
                columns = []
                for pair in images:
                    column = Image.new("RGB", (panel, 2 * panel), "white")
                    for row, image in enumerate(pair):
                        column.paste(image, (0, row * panel))
                    columns.append(column)
                frame = annotated(columns, f"Original BEFORE / adapted AFTER | t={time_s:.2f}s | top: oblique, bottom: side", [
                    f"Strike annotation: {views[0].strike:.3f}s / {views[1].strike:.3f}s; first 0.14m entry: {label(metrics[0]['first_entry_s'])} / {label(metrics[1]['first_entry_s'])}",
                    f"TRAIN snapshot {target_data['snapshot_id']}; unchanged head XYZ={np.round(target, 4).tolist()}; yellow r=0.06m",
                    "Native FK, 1x playback. Non-solid head marker; gloves r=0.08m. Terminal poses held; no dynamics."])
                writers[2].append_data(np.asarray(frame))
                if frame_id in key_indices:
                    key_tiles.append(frame.resize((480, 528)))
                if frame_id in dense_indices:
                    dense = annotated([images[0][1].resize((320, 320)), images[1][1].resize((320, 320))],
                                      f"Before / after side | frame {frame_id} | t={time_s:.2f}s",
                                      ["Actual FK render; unchanged target."])
                    dense_tiles.append(dense)
    finally:
        for renderer in renderers:
            renderer.close()
    sheet(key_tiles, 2, output / "comparison_contact_sheet.jpg")
    sheet(dense_tiles, 3, output / "comparison_dense_sheet.jpg")
    report = {"scope": "Offline kinematic reference comparison only, no stepping, policy or target collision physics",
              "target": target_data, "target_json_sha256": sha256(target_json),
              "model_sha256": sha256(H2_XML), "before": metrics[0], "after": metrics[1],
              "frames_per_video": len(times), "fps": fps, "frame_times_s": times.tolist(),
              "dense_sheet_frame_ids": sorted(dense_indices), "keyframe_ids": sorted(key_indices),
              "terminal_display": "Each stored reference holds its own terminal pose after its own end time; common video ends after the longer configured terminal hold.",
              "outputs": ["before.mp4", "after.mp4", "comparison.mp4", "comparison_contact_sheet.jpg", "comparison_dense_sheet.jpg"]}
    (output / "render_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "frames_per_video": len(times), "before": metrics[0], "after": metrics[1]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--target-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.before, args.after, args.target_json, args.output)
