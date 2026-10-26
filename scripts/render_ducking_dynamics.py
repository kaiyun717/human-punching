"""Replay recorded torque-driven ducking and its planned reference side by side."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from render_ducking import FOOTER, HEADER, PANEL, camera, font, grid, make_scene


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    ROOT / '.venv-task10c/lib/python3.12/site-packages/mjlab/'
    'asset_zoo/robots/unitree_g1/xmls/g1.xml'
)


def nearest_indices(times: np.ndarray, requested: np.ndarray) -> np.ndarray:
    """Use recorded states without interpolating across contact events."""
    right = np.searchsorted(times, requested).clip(0, len(times)-1)
    left = (right-1).clip(0)
    return np.where(abs(times[left]-requested) <= abs(times[right]-requested), left, right)


def render(trajectory_path: Path, output_dir: Path, *, model_path: Path = DEFAULT_MODEL,
           fps: int = 30, slow_motion: bool = True) -> dict:
    root = ROOT.resolve()
    source = trajectory_path.resolve(strict=True)
    output = output_dir.resolve()
    if not output.is_relative_to(root):
        raise ValueError('All video outputs must remain inside human-punching')
    if fps < 1:
        raise ValueError('Video frame rate must be positive')
    with np.load(source, allow_pickle=False) as packet:
        times = np.asarray(packet['time_s'], dtype=float)
        qpos = np.asarray(packet['qpos'], dtype=float)
        reference = np.asarray(packet['qpos_reference'], dtype=float)
        torque = np.asarray(packet['torque'], dtype=float) if 'torque' in packet else None
    if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError('Expected at least two strictly increasing simulation timestamps')
    if qpos.shape != reference.shape or qpos.shape != (len(times), 36):
        raise ValueError('Actual and reference G1 qpos must both have shape (T,36)')
    if not all(np.isfinite(value).all() for value in (times, qpos, reference)):
        raise ValueError('Cannot render a nonfinite trajectory')
    for values in (qpos, reference):
        if np.max(abs(np.linalg.norm(values[:, 3:7], axis=1)-1)) > 1e-5:
            raise ValueError('Free-base rotations must be unit quaternions')
    output.mkdir(parents=True, exist_ok=True)
    model, actual_data = make_scene(str(model_path.resolve(strict=True)))
    reference_data = mujoco.MjData(model)
    if model.nq != 36:
        raise ValueError('The visual model must be the same 29-DoF G1 layout')
    options = mujoco.MjvOption()
    options.geomgroup[:] = 0
    options.geomgroup[0] = options.geomgroup[2] = 1
    options.sitegroup[:] = 0
    # A shared fixed frame makes drift and tracking disagreement visible.
    center = np.r_[np.median(reference[:, :2], axis=0), .65]
    displacement = float(np.max(np.linalg.norm(qpos[:, :2]-center[:2], axis=1)))
    distance = max(2.2, 2.2+1.2*max(displacement-.25, 0.))
    three_quarter = camera(center, distance, 35)
    side = camera(center, distance, 90)
    title_font, text_font, small_font = font(29), font(22), font(18)
    movie = output/'g1_ducking_dynamics.mp4'
    duration = float(times[-1]-times[0])
    speeds = [1., .5] if slow_motion else [1.]
    schedules = []
    for speed in speeds:
        requested = times[0]+np.arange(int(np.floor(duration*fps/speed))+1)*speed/fps
        requested = requested[requested <= times[-1]+1e-9]
        indices = nearest_indices(times, requested)
        # Half-second endpoint holds show the initial/final posture explicitly.
        indices = np.r_[np.full(fps//2, 0), indices, np.full(fps//2, len(times)-1)]
        schedules.append((speed, indices))
    # Avoid an unbounded frame cache when long dynamics trials are supplied.
    last_index, last_panels = None, None
    renderer = mujoco.Renderer(model, PANEL, PANEL, max_geom=1000)

    def make_frame(index: int, speed: float) -> np.ndarray:
        nonlocal last_index, last_panels
        if index != last_index:
            reference_data.qpos[:] = reference[index]
            reference_data.qvel[:] = 0.
            actual_data.qpos[:] = qpos[index]
            actual_data.qvel[:] = 0.
            mujoco.mj_forward(model, reference_data)
            mujoco.mj_forward(model, actual_data)
            panels = []
            for state, view in ((reference_data, three_quarter),
                                (actual_data, three_quarter), (actual_data, side)):
                renderer.update_scene(state, camera=view, scene_option=options)
                grid(renderer.scene)
                panels.append(renderer.render().copy())
            last_panels = np.concatenate(panels, axis=1)
            last_index = index
        canvas = Image.new('RGB', (3*PANEL, HEADER+PANEL+FOOTER), (17, 23, 33))
        canvas.paste(Image.fromarray(last_panels), (0, HEADER))
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 15), 'G1 ducking / inverse-dynamics tracking', font=title_font, fill='white')
        draw.text((1610, 21), f'{speed:g}x playback', font=text_font, fill=(142, 214, 255))
        labels = ('Planned reference / kinematic', 'Torque-driven simulation', 'Simulation / side view')
        for column, label in enumerate(labels):
            draw.text((column*PANEL+24, 66), label, font=text_font, fill=(215, 226, 240))
        draw.text((24, HEADER+PANEL+17), f'Simulation time {times[index]:.3f} s',
                  font=small_font, fill=(215, 226, 240))
        error = np.linalg.norm(qpos[index, :3]-reference[index, :3])*100
        draw.text((PANEL+24, HEADER+PANEL+17), f'Pelvis position error: {error:.1f} cm',
                  font=small_font, fill=(215, 226, 240))
        label = 'Recorded forward-simulation states'
        if torque is not None and torque.ndim == 2 and len(torque) in (len(times), len(times)-1):
            value = np.max(abs(torque[min(index, len(torque)-1)]))
            label = f'Largest joint torque: {value:.1f} Nm'
        draw.text((2*PANEL+24, HEADER+PANEL+17), label, font=small_font, fill=(215, 226, 240))
        return np.asarray(canvas)

    frames = 0
    try:
        with imageio.get_writer(movie, fps=fps, codec='libx264', quality=8,
                                macro_block_size=1, pixelformat='yuv420p',
                                ffmpeg_params=['-movflags', '+faststart']) as writer:
            for speed, indices in schedules:
                for index in indices:
                    writer.append_data(make_frame(int(index), speed))
                    frames += 1
        preview_index = int(np.argmin(reference[:, 2]))
        Image.fromarray(make_frame(preview_index, 1.)).save(output/'dynamics_preview.png')
    finally:
        renderer.close()
    reader = imageio.get_reader(movie)
    try:
        decoded = sum(1 for _ in reader)
    finally:
        reader.close()
    if decoded != frames:
        raise RuntimeError(f'Video decode mismatch: {decoded} != {frames}')
    result = {
        'video': str(movie), 'source_trajectory': str(source),
        'visual_model_xml': str(model_path.resolve()), 'fps': fps,
        'frames': frames, 'width': 3*PANEL, 'height': HEADER+PANEL+FOOTER,
        'video_duration_s': frames/fps, 'simulation_duration_s': duration,
        'playback_speeds': speeds, 'state_sampling': 'nearest recorded timestamp; no interpolated physics',
        'all_frames_decoded': True, 'renderer_runs_dynamics': False,
    }
    (output/'dynamics_video.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--no-slow-motion', action='store_true')
    args = parser.parse_args()
    render(args.trajectory, args.output, model_path=args.model, fps=args.fps,
           slow_motion=not args.no_slow_motion)


if __name__ == '__main__':
    main()
