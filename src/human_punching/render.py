"""Native MuJoCo rendering, with informational overlays added after simulation."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TITLES = {"jab": "JAB", "cross": "CROSS", "left_hook": "LEFT HOOK", "right_hook": "RIGHT HOOK"}


def font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return (
        ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size=size)
    )


def render_clip(
    model,
    trajectory,
    metrics,
    output: Path,
    height_m: float,
    fps: int = 30,
    panel_width: int = 640,
    panel_height: int = 600,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    punch = metrics["source"]["punch"]
    data = mujoco.MjData(model)
    qpos, times = trajectory["qpos"], trajectory["time_s"]
    positions = trajectory["body_pos_w"]
    active = int(metrics["source"]["active_hand_joint_index"])
    root_center = np.median(positions[:, 0], axis=0)
    cameras = []
    for azimuth in (215, 110):
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [root_center[0], root_center[1], height_m * 0.48]
        camera.distance = height_m * 1.9
        camera.azimuth = azimuth
        camera.elevation = -10
        cameras.append(camera)
    renderer = mujoco.Renderer(model, height=panel_height, width=panel_width)
    options = mujoco.MjvOption()
    options.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    title_font, text_font, small_font = font(30), font(18), font(15)
    duration = float(times[-1])
    frame_cache = {}

    def draw_frame(index: int, speed_label: str) -> Image.Image:
        # Native simulator images are reused between normal and slow replay.
        if index not in frame_cache:
            data.qpos[:] = qpos[index]
            data.qvel[:] = trajectory["qvel"][index]
            data.time = float(times[index])
            mujoco.mj_forward(model, data)
            panes = []
            for camera in cameras:
                renderer.update_scene(data, camera=camera, scene_option=options)
                # A short wrist trail is a visualization aid, not a collision object.
                for t in range(max(0, index - 24), index + 1, 3):
                    scene = renderer.scene
                    if scene.ngeom >= scene.maxgeom:
                        break
                    alpha = 0.15 + 0.55 * (t - max(0, index - 24) + 1) / 25
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.008, 0, 0]),
                        positions[t, active],
                        np.eye(3).ravel(),
                        np.array([1.0, 0.78, 0.22, alpha]),
                    )
                    scene.ngeom += 1
                panes.append(Image.fromarray(renderer.render().copy()))
            frame_cache[index] = panes
        canvas = Image.new("RGB", (panel_width * 2, panel_height + 120), (15, 22, 32))
        for i, pane in enumerate(frame_cache[index]):
            canvas.paste(pane, (i * panel_width, 70))
        draw = ImageDraw.Draw(canvas)
        draw.text((25, 12), TITLES[punch], font=title_font, fill=(241, 246, 252))
        hand = "LEFT" if active == 8 else "RIGHT"
        subtitle = f"{height_m:.2f} m HUMAN   /   {hand} HAND   /   DUOBOX 120 Hz"
        draw.text((285, 24), subtitle, font=text_font, fill=(181, 197, 215))
        draw.text((panel_width * 2 - 185, 24), speed_label, font=text_font, fill=(248, 203, 80))
        for i, label in enumerate(("OBLIQUE VIEW", "SIDE VIEW")):
            x = i * panel_width + 18
            draw.rounded_rectangle((x - 7, 83, x + 153, 110), radius=4, fill=(20, 29, 40))
            draw.text((x, 88), label, font=small_font, fill=(230, 238, 247))
        footer_y = panel_height + 84
        draw.text(
            (25, footer_y),
            "MuJoCo 3.11  |  Kinematic retargeting  |  Fixed body proportions",
            font=small_font,
            fill=(174, 190, 208),
        )
        draw.text(
            (panel_width * 2 - 350, footer_y),
            f"Source frame {int(trajectory['source_frames'][index])}   t = {times[index]:.3f} s",
            font=small_font,
            fill=(205, 216, 230),
        )
        draw.line(
            (
                0,
                panel_height + 116,
                panel_width * 2 * index / max(len(times) - 1, 1),
                panel_height + 116,
            ),
            fill=(236, 180, 67),
            width=4,
        )
        return canvas

    video = output / f"{punch}.mp4"
    total_frames = 0
    try:
        with imageio.get_writer(
            video,
            fps=fps,
            codec="libx264",
            quality=8,
            ffmpeg_params=["-movflags", "+faststart"],
            macro_block_size=16,
        ) as writer:
            for speed, label in ((1.0, "1.0x REAL TIME"), (0.5, "0.5x SLOW")):
                indices = np.minimum(
                    np.round(
                        np.arange(0, duration + 1e-9, speed / fps) * float(trajectory["fps"])
                    ).astype(int),
                    len(times) - 1,
                )
                for index in indices:
                    writer.append_data(np.asarray(draw_frame(int(index), label)))
                    total_frames += 1
                for _ in range(round(fps * 0.3)):
                    writer.append_data(np.asarray(draw_frame(len(times) - 1, label)))
                    total_frames += 1
        source = metrics["source"]
        outward = source.get(
            "outward_completion_frame",
            source.get("phase_landmarks", {}).get(
                "outward_completion_frame", int(trajectory["source_frames"][len(times) // 2])
            ),
        )
        if isinstance(outward, dict):
            outward = outward.get("source_frame", int(trajectory["source_frames"][len(times) // 2]))
        still_index = int(np.argmin(np.abs(trajectory["source_frames"] - int(outward))))
        still = output / f"{punch}.png"
        draw_frame(still_index, "SOURCE MOTION").save(still)
    finally:
        renderer.close()
    return {
        "video": str(video),
        "still": str(still),
        "video_frames": total_frames,
        "video_fps": fps,
        "video_duration_s": total_frames / fps,
        "video_resolution": [panel_width * 2, panel_height + 120],
        "render_engine": "mujoco.Renderer (native OpenGL)",
        "playback_passes": [1.0, 0.5],
        "end_hold_s_per_pass": 0.3,
    }


def make_overview(stills: list[Path], output: Path) -> None:
    """A compact visual index of the four generated movies."""
    thumbs = []
    for path in stills:
        with Image.open(path) as source:
            thumbs.append(source.convert("RGB").resize((960, 540), Image.Resampling.LANCZOS))
    result = Image.new("RGB", (1920, 1080), (15, 22, 32))
    for index, thumb in enumerate(thumbs):
        result.paste(thumb, ((index % 2) * 960, (index // 2) * 540))
    result.save(output)
