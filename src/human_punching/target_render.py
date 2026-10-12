"""Native MuJoCo videos of head-targeted human motion beside a held G1.

Rendering only: the caller provides complete combined-model states.  Neither
this module nor its visual annotations step physics or change the trajectories.
"""

from __future__ import annotations

import os
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .render import TITLES, font


def _playback_indices(times: np.ndarray, speed: float, fps: int) -> np.ndarray:
    """Select the nearest source states without assuming an exact source rate."""
    sample_times = np.arange(float(times[0]), float(times[-1]) + 1e-9, speed / fps)
    right = np.clip(np.searchsorted(times, sample_times), 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(
        np.abs(times[left] - sample_times) <= np.abs(times[right] - sample_times), left, right
    )


def render_target_clip(
    model,
    trajectory: dict,
    metrics: dict,
    output: Path,
    scene_metadata: dict,
    fps: int = 30,
) -> dict:
    """Render oblique/side views, normal/slow replay, and contact/guard stills.

    Required trajectory arrays are ``qpos``, ``qvel``, ``time_s`` and
    ``active_glove_pos_w``.  They describe the entire assembled MuJoCo model.
    The clearance annotation is shown only when supplied by the retargeter;
    renderer geometry and trail markers are not used to calculate contact.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    times = np.asarray(trajectory["time_s"], dtype=float)
    qpos = np.asarray(trajectory["qpos"])
    qvel = np.asarray(trajectory["qvel"])
    glove_positions = np.asarray(trajectory["active_glove_pos_w"])
    if times.ndim != 1 or len(times) < 1 or np.any(np.diff(times) <= 0):
        raise ValueError("time_s must contain strictly increasing frame times")
    if qpos.shape != (len(times), model.nq) or qvel.shape != (len(times), model.nv):
        raise ValueError("qpos/qvel must describe every combined-model frame")
    if glove_positions.shape != (len(times), 3):
        raise ValueError("active_glove_pos_w must have shape (frames, 3)")
    if fps <= 0:
        raise ValueError("video fps must be positive")

    source = metrics.get("source", {})
    punch = metrics.get("punch", source.get("punch", "punch"))
    contact_index = int(
        np.clip(metrics.get("contact_frame_index", len(times) // 2), 0, len(times) - 1)
    )
    source_frames = np.asarray(trajectory.get("source_frames", np.arange(len(times))))
    clearances = trajectory.get("target_distance_m")
    if clearances is not None:
        clearances = np.asarray(clearances)
        if clearances.shape != times.shape:
            raise ValueError("target_distance_m must have one value per frame")
    height = float(scene_metadata.get("human_height_m", 1.8))
    data = mujoco.MjData(model)
    panel_width, panel_height = 640, 600

    # The midpoint of body origins keeps both actors in frame, including models
    # whose free-joint order differs from the reference mannequin's ordering.
    actor_positions = []
    for index in sorted({0, contact_index, len(times) - 1}):
        data.qpos[:] = qpos[index]
        data.qvel[:] = qvel[index]
        mujoco.mj_forward(model, data)
        actor_positions.append(data.xpos[1:].copy())
    positions = np.concatenate(actor_positions, axis=0)
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    center = (lo + hi) / 2
    center[2] = max(height, hi[2] + 0.12) * 0.48
    span_xy = float(np.linalg.norm(hi[:2] - lo[:2]))
    distance = max(height * 1.72, span_xy * 1.60)
    cameras = []
    for azimuth, elevation in ((225, -11), (90, -6)):
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = center
        camera.distance = distance
        camera.azimuth = azimuth
        camera.elevation = elevation
        cameras.append(camera)

    options = mujoco.MjvOption()
    options.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    renderer = mujoco.Renderer(model, height=panel_height, width=panel_width)
    title_font, text_font, small_font = font(29), font(17), font(14)
    frame_cache: dict[int, list[Image.Image]] = {}

    def draw_frame(index: int, speed_label: str) -> Image.Image:
        if index not in frame_cache:
            data.qpos[:] = qpos[index]
            data.qvel[:] = qvel[index]
            data.time = float(times[index])
            mujoco.mj_forward(model, data)
            panes = []
            # A 0.18 s trail displays trajectory direction without obscuring the
            # target.  All these geoms live only in the rendering scene.
            first = int(np.searchsorted(times, times[index] - 0.18))
            trail_indices = np.unique(
                np.linspace(first, index, min(9, index - first + 1)).astype(int)
            )
            for camera in cameras:
                renderer.update_scene(data, camera=camera, scene_option=options)
                for number, frame_index in enumerate(trail_indices):
                    scene = renderer.scene
                    if scene.ngeom >= scene.maxgeom:
                        break
                    alpha = 0.10 + 0.45 * (number + 1) / max(len(trail_indices), 1)
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.005, 0, 0]),
                        glove_positions[frame_index],
                        np.eye(3).ravel(),
                        np.array([1.0, 0.78, 0.22, alpha]),
                    )
                    scene.ngeom += 1
                panes.append(Image.fromarray(renderer.render().copy()))
            frame_cache[index] = panes

        canvas = Image.new("RGB", (1280, 720), (15, 22, 32))
        for pane_index, pane in enumerate(frame_cache[index]):
            canvas.paste(pane, (pane_index * panel_width, 70))
        draw = ImageDraw.Draw(canvas)
        draw.text((22, 13), TITLES.get(punch, punch.upper()), font=title_font, fill=(241, 246, 252))
        draw.text(
            (255, 24),
            f"{height:.2f} m HUMAN  /  STANDING UNITREE G1",
            font=text_font,
            fill=(181, 197, 215),
        )
        draw.text((1078, 24), speed_label, font=text_font, fill=(248, 203, 80))
        for pane_index, label in enumerate(("OBLIQUE VIEW", "SIDE VIEW")):
            x = pane_index * panel_width + 18
            draw.rounded_rectangle((x - 7, 82, x + 143, 110), radius=4, fill=(20, 29, 40))
            draw.text((x, 88), label, font=small_font, fill=(230, 238, 247))
        # Put numerical target evidence beside the corresponding freeze frame,
        # with its exact interpretation left to the accompanying metrics file.
        if index == contact_index:
            badge = "CLOSEST APPROACH"
            draw.rounded_rectangle((477, 627, 797, 657), radius=5, fill=(23, 42, 54))
            draw.text((556, 634), badge, font=small_font, fill=(236, 207, 105))
        draw.text(
            (22, 682),
            "KINEMATIC IK PREVIEW  |  G1 POSE HELD  |  NO IMPACT DYNAMICS",
            font=small_font,
            fill=(182, 198, 215),
        )
        detail = f"Source {int(source_frames[index])}   t = {times[index] - times[0]:.3f} s"
        draw.text((945, 682), detail, font=small_font, fill=(205, 216, 230))
        draw.line(
            (0, 716, 1280 * index / max(len(times) - 1, 1), 716), fill=(236, 180, 67), width=4
        )
        return canvas

    video = output / f"{punch}.mp4"
    contact_still = output / f"{punch}-contact.png"
    guard_still = output / f"{punch}-guard.png"
    total_frames = 0
    end_hold_frames = round(fps * 0.30)
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
                for index in _playback_indices(times, speed, fps):
                    writer.append_data(np.asarray(draw_frame(int(index), label)))
                    total_frames += 1
                for _ in range(end_hold_frames):
                    writer.append_data(np.asarray(draw_frame(len(times) - 1, label)))
                    total_frames += 1
        draw_frame(contact_index, "CLOSEST FRAME").save(contact_still)
        draw_frame(0, "INITIAL GUARD").save(guard_still)
    finally:
        renderer.close()
    return {
        "video": str(video),
        "still": str(contact_still),
        "contact_still": str(contact_still),
        "guard_still": str(guard_still),
        "contact_frame_index": contact_index,
        "video_frames": total_frames,
        "video_fps": fps,
        "video_duration_s": total_frames / fps,
        "video_resolution": [1280, 720],
        "render_engine": "mujoco.Renderer (native OpenGL)",
        "requested_gl_backend": os.environ.get("MUJOCO_GL", "default"),
        "playback_passes": [1.0, 0.5],
        "end_hold_s_per_pass": end_hold_frames / fps,
        "camera_azimuth_deg": [225, 90],
        "camera_distance_m": distance,
        "camera_lookat_w": center.tolist(),
        "cached_source_frames": len(frame_cache),
        "visual_trail_duration_s": 0.18,
        "physical_stepping": False,
    }
