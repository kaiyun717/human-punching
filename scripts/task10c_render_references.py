"""Render stored Task10C reference previews; use inside an allocated GPU job."""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from human_punching.task10c.references import StoredReference
from human_punching.task10c.robot import H2_XML, ROOT


def render(family: str, reference_dir: Path, output: Path):
    ref = StoredReference.load(reference_dir / f"{family}.npz")
    spec = mujoco.MjSpec.from_file(str(H2_XML))
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[8, 8, .1], rgba=[.82, .84, .87, 1])
    spec.worldbody.add_light(pos=[2, -2, 4], dir=[-1, 1, -2], diffuse=[.8, .8, .8])
    model, data = spec.compile(), None
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    renderer = mujoco.Renderer(model, height=600, width=600)
    cameras = []
    for azimuth in (145, 90):
        cam = mujoco.MjvCamera()
        cam.distance = 3.6
        cam.azimuth, cam.elevation = azimuth, -12
        cam.lookat[:] = [.15, 0, .95]
        cameras.append(cam)
    opts = mujoco.MjvOption()
    opts.geomgroup[2] = 1
    opts.geomgroup[3] = 1
    # Show glove collision spheres, but hide interior primitive skeleton shapes.
    for gid in range(model.ngeom):
        name = model.geom(gid).name or ""
        if name.endswith("_collision") and "glove" not in name:
            model.geom_rgba[gid, 3] = 0
    selected = np.linspace(0, len(ref.time_s)-1, 8).round().astype(int)
    tiles = []
    output.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output / f"{family}.mp4", fps=50, codec="libx264", quality=7) as writer:
        for i, q in enumerate(ref.arrays["qpos"]):
            data.qpos[:] = q
            data.qvel[:] = ref.arrays["qvel"][i]
            mujoco.mj_forward(model, data)
            views = []
            for cam in cameras:
                renderer.update_scene(data, cam, scene_option=opts)
                views.append(renderer.render().copy())
            frame = Image.fromarray(np.concatenate(views, axis=1))
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 1200, 35), fill=(245, 245, 245))
            draw.text((10, 10), f"{family}: source {ref.arrays['source_frames'][i]:.1f} | t={ref.time_s[i]:.2f}s | nominal strike={ref.metadata['strike_time_s']:.3f}s | red=left, blue=right | kinematic reference", fill=(0, 0, 0))
            writer.append_data(np.asarray(frame))
            if i in selected:
                tiles.append(frame.resize((600, 300)))
    sheet = Image.new("RGB", (1200, 1200), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 2)*600, (i//2)*300))
    sheet.save(output / f"{family}_contact_sheet.jpg", quality=90)
    renderer.close()
    print(f"rendered {family}: {len(ref.time_s)} frames -> {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", default=["jab", "left_hook"])
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "outputs/task10c/references")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/task10c/references/previews")
    args = parser.parse_args()
    for family in args.families:
        render(family, args.reference_dir, args.output)
