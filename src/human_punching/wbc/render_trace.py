"""Render recorded diagnostic states on an allocated GPU; never integrate physics.

The recorded qpos is assigned only to a separate visualization model. These
assignments are playback, not resets or new evidence about physical execution.
The first environment for each family is preselected before rendering.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np

from .provenance import sha256_file, writable_path, write_json


def render_recorded_cycles(source: Path, output: Path) -> dict:
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Recorded-state rendering requires an allocated GPU")
    os.environ["MUJOCO_GL"] = "egl"
    source = source.resolve(strict=True)
    audit = json.loads((source / "audit.json").read_text())
    if audit.get("schema") != "human_wbc_native_cycle_probe_v1":
        raise ValueError("Expected a complete recorded cycle diagnostic")
    if audit.get("raw_trace_sha256") != sha256_file(source / "raw_trace.npz"):
        raise ValueError("Recorded trace hash mismatch")
    compiled = audit["compiled_scene"]
    model_path = source / "compiled_scene.mjb"
    if compiled["sha256"] != sha256_file(model_path):
        raise ValueError("Compiled visualization model differs from the executed scene")
    requests = json.loads((source / "launch_requests.json").read_text())
    families = ("jab", "cross", "left_hook", "right_hook")
    if tuple(requests[i]["family"] for i in range(4)) != families:
        raise ValueError("Expected deterministic first four family assignments")
    destination = writable_path(output)
    destination.mkdir(parents=True, exist_ok=False)
    for variable, child in (("XDG_CACHE_HOME", "xdg"), ("CUDA_CACHE_PATH", "cuda"),
                            ("TMPDIR", "tmp"), ("MPLCONFIGDIR", "matplotlib")):
        cache = destination / "cache" / child
        cache.mkdir(parents=True)
        os.environ[variable] = str(cache)
    from .egl_binding import configure_allocated_egl
    binding = configure_allocated_egl()
    write_json(destination / "gpu_binding.json", binding)
    import mujoco
    import imageio.v2 as imageio
    import imageio_ffmpeg
    from PIL import Image, ImageDraw

    model = mujoco.MjModel.from_binary_path(str(model_path))
    model.vis.global_.offwidth, model.vis.global_.offheight = 800, 600
    model.vis.headlight.ambient[:] = .5
    data = mujoco.MjData(model)
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[:] = 1  # Physical human collision primitives are group 3.
    scene_option.sitegroup[:] = 0
    with np.load(source / "raw_trace.npz", allow_pickle=False) as archive:
        qpos, qvel, times = (archive[name].copy() for name in ("qpos", "qvel", "time"))
    if qpos.shape != (1201, 16, 55) or not np.isfinite(qpos).all():
        raise ValueError("Expected finite complete six-second native trace")
    frame_ids = np.arange(0, 1200, 4)  # 50 fps playback of 200 Hz measurements.
    start = time.monotonic()
    products = []
    with mujoco.Renderer(model, height=600, width=800) as renderer:
        for environment, family in enumerate(families):
            target = np.asarray(requests[environment]["snapshot"]["head_position_w"])
            camera = mujoco.MjvCamera()
            camera.lookat[:] = (qpos[0, environment, :3] + target) / 2
            camera.lookat[2] = .8
            camera.distance, camera.elevation = 3.7, -14.
            delta = target[:2] - qpos[0, environment, :2]
            camera.azimuth = np.degrees(np.arctan2(delta[1], delta[0])) + 125.
            video = destination / f"{family}_recorded_failure.mp4"
            with imageio.get_writer(video, fps=50, codec="libx264", quality=7,
                                    macro_block_size=1, ffmpeg_log_level="error") as writer:
                for index in frame_ids:
                    if time.monotonic()-start > 780:
                        raise TimeoutError("Bounded rendering exceeded thirteen minutes")
                    data.qpos[:], data.qvel[:], data.time = qpos[index, environment], qvel[index, environment], times[index, environment]
                    mujoco.mj_forward(model, data)  # FK only; no mj_step anywhere.
                    renderer.update_scene(data, camera=camera, scene_option=scene_option)
                    geom = renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.full(3, .14), target, np.eye(3).ravel(), np.array([1., .15, .1, .25]))
                    renderer.scene.ngeom += 1
                    frame = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 800, 60), fill=(15, 15, 15))
                    draw.text((12, 8), f"{family} | t={times[index, environment]:.3f}s | recorded zero-residual FAILURE", fill="white")
                    draw.text((12, 28), "Frozen target: red 0.14 m sphere. No G1 collision response. No learned actor.", fill="white")
                    draw.text((12, 44), "Playback of measured physical states; no new physics integration.", fill="white")
                    writer.append_data(np.asarray(frame))
                    if index in (0, 100, 200, 320, 600, 1196):
                        frame.save(destination / f"{family}_{index:04d}.png")
            reader = imageio.get_reader(video)
            try:
                count = sum(1 for _ in reader)
            finally:
                reader.close()
            if count != len(frame_ids):
                raise ValueError("Encoded video did not fully decode")
            products.append({"family": family, "environment": environment, "path": video.name,
                "sha256": sha256_file(video), "decoded_frames": count, "fps": 50,
                "simulated_duration_s": 6., "source_outcome": audit["episodes"][environment]["result"]})
            if sum(p.stat().st_size for p in destination.rglob("*") if p.is_file()) > 2*1024**3:
                raise RuntimeError("Rendering product cap exceeded")
    report = {"schema": "human_wbc_recorded_media_v1", "job_id": os.environ["SLURM_JOB_ID"],
        "inputs": {name: sha256_file(source / name) for name in
                   ("audit.json", "compiled_scene.mjb", "raw_trace.npz", "launch_requests.json")},
        "renderer_source_sha256": sha256_file(Path(__file__)),
        "egl_binding": binding,
        "ffmpeg_sha256": sha256_file(imageio_ffmpeg.get_ffmpeg_exe()), "products": products,
        "physics_integrated": False, "learning": False, "acceptance_eligible": False,
        "selection": "first environment for each family; all four failures retained",
        "visualization": "All geometry groups visible, including group-3 physical body; sites hidden; source model dynamics unchanged",
        "wall_seconds": time.monotonic()-start}
    write_json(destination / "manifest.json", report)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    render_recorded_cycles(arguments.source, arguments.output)
