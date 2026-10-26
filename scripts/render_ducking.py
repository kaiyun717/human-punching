"""Render the recorded DuoBox duck beside its kinematic GMR G1 retargeting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from human_punching.ducking import retarget_duck
from human_punching.motion import PARENTS


PANEL = 640
HEADER = 100
FOOTER = 60
FPS = 30


def font(size: int):
    candidates = (
        Path('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf'),
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size=size)


def make_scene(xml_path: str | None):
    spec = mujoco.MjSpec.from_file(xml_path) if xml_path else mujoco.MjSpec()
    spec.worldbody.add_geom(name='ducking_floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[8, 8, .1], rgba=[.22, .25, .29, 1],
                            contype=0, conaffinity=0)
    spec.worldbody.add_light(pos=[2, -3, 4], dir=[-.3, .3, -1],
                             diffuse=[.8, .8, .8], castshadow=True)
    spec.worldbody.add_light(pos=[-2, 2, 3], dir=[.3, -.3, -1],
                             diffuse=[.35, .35, .35], castshadow=False)
    spec.visual.global_.offwidth = PANEL
    spec.visual.global_.offheight = PANEL
    spec.visual.headlight.ambient = [.35, .35, .35]
    spec.visual.headlight.diffuse = [.65, .65, .65]
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def camera(lookat, distance: float, azimuth: float):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, -12
    return cam


def segment(scene, start, end, radius, rgba):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                       np.zeros(3), np.eye(3).ravel(), np.asarray(rgba))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, start, end)
    scene.ngeom += 1


def sphere(scene, center, radius, rgba):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                       np.full(3, radius), center, np.eye(3).ravel(), np.asarray(rgba))
    scene.ngeom += 1


def skeleton(scene, points, rgba):
    for child, parent in enumerate(PARENTS):
        if parent >= 0:
            segment(scene, points[parent], points[child], .021, rgba)
    for joint in (0, 6, 7, 10, 11, 14, 18):
        sphere(scene, points[joint], .028, rgba)
    sphere(scene, points[4], .075, rgba)
    for hand in (8, 12):
        sphere(scene, points[hand], .045, rgba)


def grid(scene):
    for coordinate in np.arange(-3, 3.01, .5):
        segment(scene, [-3, coordinate, .002], [3, coordinate, .002], .0015,
                [.4, .44, .49, 1])
        segment(scene, [coordinate, -3, .002], [coordinate, 3, .002], .0015,
                [.4, .44, .49, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(root):
        raise ValueError('Output must stay inside human-punching')
    output.mkdir(parents=True, exist_ok=True)
    if args.reuse:
        info = json.loads((output / 'retarget.json').read_text())
    else:
        info = retarget_duck(output)
        (output / 'retarget.json').write_text(json.dumps(info, indent=2) + '\n')
    packet = np.load(info['trajectory_path'])
    qpos = packet['qpos']
    source = packet['source_joint_positions']
    opponent = packet['opponent_joint_positions']
    frame_ids = packet['source_frames']
    model, data = make_scene(info['model_xml_path'])
    src_model, src_data = make_scene(None)
    if qpos.shape[1] != model.nq:
        raise ValueError('Trajectory and rendered G1 disagree')
    options = mujoco.MjvOption()
    options.geomgroup[:] = 0
    options.geomgroup[0] = options.geomgroup[2] = 1
    options.sitegroup[:] = 0
    robot_center = np.r_[np.median(qpos[:, :2], axis=0), .65]
    all_points = np.concatenate((source, opponent), axis=1)
    source_center = .5 * (all_points.reshape(-1, 3).min(0) + all_points.reshape(-1, 3).max(0))
    source_center[2] = .65
    cam_src = camera(source_center, 3.15, 110)
    cameras = [camera(robot_center, 2.2, 35), camera(robot_center, 2.2, 90)]
    target = output / 'duobox_duck_to_g1.mp4'
    renderers = [mujoco.Renderer(src_model, PANEL, PANEL, max_geom=1000),
                 mujoco.Renderer(model, PANEL, PANEL, max_geom=1000)]
    cached = {}
    title_font, text_font, small_font = font(29), font(22), font(18)

    def make_frame(index, speed):
        if index not in cached:
            data.qpos[:] = qpos[index]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            renderers[0].update_scene(src_data, camera=cam_src)
            grid(renderers[0].scene)
            skeleton(renderers[0].scene, opponent[index], [1., .43, .22, 1.])
            skeleton(renderers[0].scene, source[index], [.2, .78, 1., 1.])
            panels = [renderers[0].render().copy()]
            for cam in cameras:
                renderers[1].update_scene(data, camera=cam, scene_option=options)
                grid(renderers[1].scene)
                panels.append(renderers[1].render().copy())
            cached[index] = np.concatenate(panels, axis=1)
        canvas = Image.new('RGB', (3 * PANEL, HEADER + PANEL + FOOTER), (17, 23, 33))
        canvas.paste(Image.fromarray(cached[index]), (0, HEADER))
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 15), 'DuoBox ducking  /  GMR to Unitree G1', font=title_font, fill='white')
        draw.text((1610, 21), f'{speed} playback', font=text_font, fill=(142, 214, 255))
        for x, label in zip((24, PANEL+24, 2*PANEL+24),
                            ('Recorded human motion', 'G1 / three-quarter view', 'G1 / side view')):
            draw.text((x, 66), label, font=text_font, fill=(215, 226, 240))
        draw.text((24, HEADER+PANEL+17), 'Blue: defender   Orange: incoming punch', font=small_font,
                  fill=(215, 226, 240))
        draw.text((PANEL+24, HEADER+PANEL+17), 'Kinematic retargeting', font=small_font,
                  fill=(215, 226, 240))
        draw.text((2*PANEL+24, HEADER+PANEL+17),
                  f'Sequence 006 / source frame {int(frame_ids[index])}', font=small_font,
                  fill=(215, 226, 240))
        return np.asarray(canvas)

    count = 0
    try:
        with imageio.get_writer(target, fps=FPS, codec='libx264', quality=8,
                                macro_block_size=1, pixelformat='yuv420p',
                                ffmpeg_params=['-movflags', '+faststart']) as writer:
            for label, stride in (('1x', 4), ('0.5x', 2)):
                indices = [0]*15 + list(range(0, len(qpos), stride)) + [len(qpos)-1]*15
                for index in indices:
                    frame = make_frame(index, label)
                    writer.append_data(frame)
                    count += 1
            preview_index = int(np.argmin(source[:, 4, 2]))
            Image.fromarray(make_frame(preview_index, '0.5x')).save(output/'ducking_preview.png')
    finally:
        for renderer in renderers:
            renderer.close()
    reader = imageio.get_reader(target)
    decoded = sum(1 for _ in reader)
    reader.close()
    if decoded != count:
        raise RuntimeError(f'Video decode mismatch: {decoded} != {count}')
    (output/'video.json').write_text(json.dumps({
        'video': str(target), 'frames': count, 'fps': FPS, 'width': 3*PANEL,
        'height': HEADER+PANEL+FOOTER, 'duration_s': count/FPS,
        'source_frame_first': int(frame_ids[0]), 'source_frame_last': int(frame_ids[-1]),
        'playback_speeds': [1., .5], 'dynamics_simulated': False,
        'all_frames_decoded': True,
    }, indent=2)+'\n')
    print(f'Video complete: {target} ({count} frames)', flush=True)


if __name__ == '__main__':
    main()
