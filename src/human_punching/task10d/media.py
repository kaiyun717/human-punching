"""Small streaming diagnostic media; never modifies the simulation model/state."""
from __future__ import annotations

from copy import copy
import os
from pathlib import Path
import uuid

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


def bind_allocated_egl() -> dict:
    """Reuse accepted UUID matching without its old interpreter-path restriction."""
    import torch
    from boxing.perception.egl import _egl_cuda_devices, _select_egl_device
    if not os.environ.get("SLURM_JOB_ID") or torch.cuda.device_count() != 1:
        raise RuntimeError("External rendering requires one allocated CUDA GPU")
    value = torch.cuda.get_device_properties(0).uuid
    if isinstance(value, bytes):
        value = value.decode("ascii")
    allocated = "GPU-" + str(uuid.UUID(str(value).removeprefix("GPU-")))
    selected = _select_egl_device(_egl_cuda_devices(), allocated)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(selected["egl_device_index"])
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    return selected


def cpu(value) -> np.ndarray:
    value = getattr(value, "_tensor", value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


class ExternalRenderer:
    def __init__(self, host_model):
        self.binding = bind_allocated_egl()
        self.model = copy(host_model)
        self.model.vis.global_.offwidth = 672
        self.model.vis.global_.offheight = 376
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=376, width=672)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = (1, 1, 1, 0, 0, 0)
        self.cameras = []
        for azimuth in (130., 90.):
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance, cam.azimuth, cam.elevation = 3.6, azimuth, -13.
            cam.lookat[:] = (.55, .12, .95)
            self.cameras.append(cam)

    def render(self, sim_data, time_s: float) -> list[np.ndarray]:
        # All prescribed links are copied, not just the glove mocap bodies.
        for name in ("qpos", "qvel", "mocap_pos", "mocap_quat"):
            getattr(self.data, name)[:] = cpu(getattr(sim_data, name))[0]
        self.data.time = time_s
        mujoco.mj_forward(self.model, self.data)
        result = []
        for camera in self.cameras:
            self.renderer.update_scene(self.data, camera, scene_option=self.option)
            result.append(self.renderer.render().copy())
        return result

    def close(self):
        self.renderer.close()


def depth_panel(depth, mask):
    shade = np.where(mask, np.rint(np.clip(depth, 0, 1) * 230), 255).astype(np.uint8)
    return Image.fromarray(shade).convert("RGB").resize((672, 376), Image.Resampling.NEAREST)


class DiagnosticMedia:
    def __init__(self, output: Path, model, title: str):
        self.output, self.title = output, title
        output.mkdir(parents=True, exist_ok=False)
        self.renderer = ExternalRenderer(model)
        kwargs = dict(fps=50, codec="libx264", quality=7, macro_block_size=1,
                      ffmpeg_params=["-threads", "1"])
        self.external = imageio.get_writer(output / "external.mp4", **kwargs)
        self.perception = imageio.get_writer(output / "perception.mp4", **kwargs)
        self.count = 0
        self.tiles = []

    def append(self, sim_data, exposure, *, time_s, source_s, hazard, keyframe=False):
        views = self.renderer.render(sim_data, time_s)
        canvas = Image.new("RGB", (1344, 432), "#151b23")
        for i, view in enumerate(views):
            canvas.paste(Image.fromarray(view), (672 * i, 56))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), self.title + " | prescribed H2 + dynamic frozen Fresh96 G1", fill="white")
        draw.text((10, 28), f"t={time_s:.3f}s source={source_s:.3f}s | DIAGNOSTIC full cycle; first hazard={hazard}", fill="white")
        self.external.append_data(np.asarray(canvas))
        frame = Image.new("RGB", (1344, 848), "#151b23")
        frame.paste(Image.fromarray(exposure["rgb"][0]), (0, 48))
        frame.paste(depth_panel(exposure["acquired_normalized"][0], exposure["acquired_masks"][0]), (672, 48))
        frame.paste(depth_panel(exposure["corrupted_normalized"][0], exposure["corrupted_masks"][0]), (0, 472))
        frame.paste(depth_panel(exposure["delivered_normalized"][0], exposure["delivered_masks"][0]), (672, 472))
        draw = ImageDraw.Draw(frame)
        source_tick = int(exposure["delivered_source_tick"][0])
        delay = int(exposure["delay_ticks"][0])
        draw.text((8, 8), f"{self.title} | acquisition {time_s:.3f}s; fixed delay {20 * delay} ms", fill="white")
        draw.text((8, 28), "Actual mounted RGB 672x376", fill="white")
        draw.text((680, 28), "Acquired RGB-HSV foreground depth 16x9", fill="white")
        draw.text((8, 438), "Same acquisition + Task08 visual faults (white = invalid)", fill="white")
        draw.text((680, 438), f"Delivered depth: source tick {source_tick}; source {source_tick * .02:.3f}s", fill="white")
        draw.text((8, 456), "Fresh96 uses proprioception only; these visuals do not alter its input", fill="white")
        self.perception.append_data(np.asarray(frame))
        if keyframe:
            canvas.save(self.output / f"external_{self.count:04d}.jpg", quality=92)
            frame.save(self.output / f"perception_{self.count:04d}.png")
            Image.fromarray(exposure["rgb"][0]).save(self.output / f"rgb_{self.count:04d}.png")
            np.savez_compressed(self.output / f"depth_{self.count:04d}.npz", **{
                k: v for k, v in exposure.items() if isinstance(v, np.ndarray) and k != "rgb"})
            self.tiles.append(canvas.resize((672, 216)))
        self.count += 1

    def close(self):
        self.external.close()
        self.perception.close()
        self.renderer.close()
        if self.tiles:
            sheet = Image.new("RGB", (1344, 216 * ((len(self.tiles) + 1) // 2)), "white")
            for i, tile in enumerate(self.tiles):
                sheet.paste(tile, ((i % 2) * 672, (i // 2) * 216))
            sheet.save(self.output / "contact_sheet.jpg", quality=92)
