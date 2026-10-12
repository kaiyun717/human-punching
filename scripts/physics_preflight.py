#!/usr/bin/env python3
"""Exercise training, batched contacts, and the Task 08 Warp camera API on CUDA.

Each stage is recorded separately; a render-only pass cannot mask a training failure.
The optional scene probe checks real mannequin/G1 mesh assets without claiming control.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
import traceback
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--trajectory', type=Path)
    parser.add_argument('--physical-model', type=Path)
    parser.add_argument('--worlds', type=int, default=16)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'status': 'running', 'job_id': os.environ.get('SLURM_JOB_ID'),
        'hostname': platform.node(), 'python': platform.python_version(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'mjlab_git_revision': '8ee51fbcf806a7419189f706d9e394cbeb7790fa',
        'packages': {}, 'stages': {},
    }
    for package in ('torch', 'mujoco', 'mujoco-warp', 'warp-lang', 'mjlab', 'numpy', 'rsl-rl-lib'):
        try:
            report['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report['packages'][package] = None

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    def stage(name, fn):
        print(f'START {name}', flush=True)
        started = time.perf_counter()
        try:
            value = fn()
            report['stages'][name] = {'status': 'passed', 'elapsed_s': time.perf_counter()-started, **value}
        except Exception as error:
            report['stages'][name] = {'status': 'failed', 'elapsed_s': time.perf_counter()-started,
                                      'error': str(error), 'traceback': traceback.format_exc()}
        save()
        print(json.dumps({name: report['stages'][name]}), flush=True)

    if not report['job_id'] or not report['cuda_visible_devices']:
        report['status'] = 'failed'
        report['error'] = 'Run through a Slurm GPU allocation; no CPU fallback.'
        save()
        return 2

    def torch_train():
        import torch
        assert torch.cuda.is_available()
        report['torch_arch_list'] = torch.cuda.get_arch_list()
        report['torch_cuda_build'] = torch.version.cuda
        report['gpu'] = torch.cuda.get_device_name(0)
        report['compute_capability'] = list(torch.cuda.get_device_capability(0))
        save()
        torch.manual_seed(991)
        model = torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.ELU(), torch.nn.Linear(128, 29)).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(256, 128, device='cuda')
        y = torch.randn(256, 29, device='cuda')
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            loss = (model(x) - y).square().mean()
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
            losses.append(float(loss.detach()))
        torch.cuda.synchronize()
        assert losses[-1] < losses[0]
        return {'losses': losses, 'finite_gradients': True, 'updates': 5}

    def warp_kernel():
        import warp as wp
        wp.init()
        wp.config.kernel_cache_dir = str(Path('.cache/warp').resolve())
        device = wp.get_device('cuda:0')
        assert device.is_cuda
        source = wp.array(np.arange(512, dtype=np.float32), device=device)
        destination = wp.zeros_like(source)
        kernel = wp.Kernel(func=_double_kernel, key='human_punching_double')
        wp.launch(kernel, dim=512, inputs=[source, destination], device=device)
        wp.synchronize_device(device)
        np.testing.assert_allclose(destination.numpy(), np.arange(512, dtype=np.float32)*2)
        return {'device': str(device), 'architecture': device.arch, 'count': 512,
                'cuda_toolkit_version': str(wp.get_cuda_toolkit_version())}

    def batched_contacts():
        import mujoco
        import torch
        import warp as wp
        from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
        xml = '''<mujoco><option timestep="0.005" integrator="implicitfast"/>
          <worldbody><light pos="0 0 3"/><geom name="floor" type="plane" size="3 3 .1"/>
            <body name="ball" pos="0 0 .7"><freejoint/><geom name="ball_geom" type="sphere"
               size=".1" mass="1" rgba="1 0 0 1" friction=".8 .01 .001"/></body>
            <camera name="overhead" pos="0 0 3" quat="1 0 0 0" fovy="45"/>
          </worldbody></mujoco>'''
        model = mujoco.MjModel.from_xml_string(xml)
        sim = Simulation(args.worlds, SimulationCfg(nconmax=8, njmax=32,
            mujoco=MujocoCfg(timestep=.005, iterations=30, ls_iterations=10)), model, 'cuda:0')
        with wp.ScopedDevice(sim.wp_device):
            stream = torch.cuda.ExternalStream(wp.get_stream().cuda_stream)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            sim.data.qpos[:, 2] = torch.linspace(.4, .8, args.worlds, device='cuda')
            initial = sim.data.qpos.clone()
            sim.forward()
            for _ in range(240):
                sim.step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        qpos = sim.data.qpos.cpu().numpy()
        qvel = sim.data.qvel.cpu().numpy()
        assert np.isfinite(qpos).all() and np.isfinite(qvel).all()
        assert np.all((qpos[:, 2] > .09) & (qpos[:, 2] < .11)), qpos[:, 2]
        assert np.max(np.abs(qvel)) < .01
        camera = _render_camera(sim.mj_model, sim.wp_model, sim.wp_data, args.output.parent/'fixture-camera', 160, 120)
        return {'worlds': args.worlds, 'steps': 240, 'simulated_seconds': 1.2,
                'initial_z_range_m': [float(initial[:, 2].min()), float(initial[:, 2].max())],
                'final_z_range_m': [float(qpos[:, 2].min()), float(qpos[:, 2].max())],
                'max_abs_velocity': float(np.max(np.abs(qvel))),
                'gravity_and_floor_contact_verified': True, 'cuda_graph': sim.use_cuda_graph, 'camera': camera}

    def actual_scene():
        import mujoco
        import mujoco_warp as mjwarp
        import warp as wp
        spec = mujoco.MjSpec.from_file(str(args.scene.resolve()))
        # A camera above/sideways sees both existing entities; pure Warp, no OpenGL.
        spec.worldbody.add_camera(name='runtime_probe_camera', pos=[1., -3.2, 1.7],
                                  xyaxes=[1, 0, 0, 0, .22, .975], fovy=45)
        # Match Task 08 rectified_camera.py/camera.py, adapting only the body prefix.
        half_pitch = math.radians(15.) / 2
        c, s = math.cos(half_pitch), math.sin(half_pitch)
        spec.body('g1/torso_link').add_camera(name='task08_runtime_probe',
            pos=[.08, 0., .45], quat=[(c+s)/2, (c-s)/2, (-c+s)/2, (-c-s)/2],
            fovy=54., resolution=[672,376])
        model = spec.compile()
        data = mujoco.MjData(model)
        frame = None
        if args.trajectory:
            trajectory = np.load(args.trajectory)
            frame = len(trajectory['qpos']) // 2
            assert trajectory['qpos'].shape[1] == model.nq
            data.qpos[:] = trajectory['qpos'][frame]
        mujoco.mj_forward(model, data)
        with wp.ScopedDevice('cuda:0'):
            wm = mjwarp.put_model(model)
            wd = mjwarp.put_data(model, data, nworld=2, nconmax=16, njmax=128)
            mjwarp.forward(wm, wd)
            wp.synchronize()
            result = _render_camera(model, wm, wd, args.output.parent/'actual-scene-camera', 320, 240,
                                    camera_id=model.camera('runtime_probe_camera').id)
            task08 = _render_camera(model, wm, wd, args.output.parent/'task08-camera', 672, 376,
                                    camera_id=model.camera('task08_runtime_probe').id)
            assert 'Left_boxing_glove' in task08['visible_geom_names'] or 'Right_boxing_glove' in task08['visible_geom_names'], 'No glove visible in the Task 08 camera.'
        return {'scene': str(args.scene), 'xml_sha256': hashlib.sha256(args.scene.read_bytes()).hexdigest(),
                'trajectory_frame': frame, 'nq': model.nq, 'nv': model.nv, 'nmesh': model.nmesh,
                'ngeom': model.ngeom, 'worlds': 2, 'camera': result, 'task08_camera': task08,
                'task08_camera_parameters': {'raster':[672,376], 'fovy_deg':54., 'parent':'g1/torso_link',
                    'mount_position_m':[.08,0.,.45], 'pitch_down_deg':15.,
                    'scope':'Matched raster/extrinsics and raw RGB/depth/IDs; no HSV LUT, pooling, delay, or policy inference.'},
                'scope': 'Existing mannequin/G1 geometry + kinematic FK + Warp RGB/depth/IDs; not physical control.'}

    def physical_human():
        import mujoco
        import torch
        import warp as wp
        from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
        path = args.physical_model.resolve()
        model = mujoco.MjModel.from_xml_path(str(path))
        assert (model.nq, model.nv, model.nu) == (87, 66, 60)
        assert np.sum(model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE)) == 1
        assert np.sum(model.jnt_type == int(mujoco.mjtJoint.mjJNT_BALL)) == 20
        # Direct control, no scalar-joint PD actuator wrapper or root wrench.
        sim = Simulation(4, SimulationCfg(nconmax=128, njmax=512,
            mujoco=MujocoCfg(timestep=.002, cone='elliptic', iterations=60,
                            ls_iterations=20, tolerance=1e-9)), model, 'cuda:0')
        with wp.ScopedDevice(sim.wp_device):
            stream = torch.cuda.ExternalStream(wp.get_stream().cuda_stream)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            sim.data.ctrl.zero_()
            sim.data.ctrl[1:, 0] = .5
            for _ in range(100):
                sim.step()
            sim.forward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        qpos, qvel = sim.data.qpos.cpu().numpy(), sim.data.qvel.cpu().numpy()
        assert np.isfinite(qpos).all() and np.isfinite(qvel).all()
        response = float(np.max(np.abs(qvel[1]-qvel[0])))
        assert response > 1e-6, 'The ball-joint torque did not change the dynamics.'
        quaternion_norm_error = float(np.max(np.abs(np.linalg.norm(qpos[:,7:].reshape(4,20,4),axis=2)-1)))
        assert quaternion_norm_error < 1e-4
        camera = _render_camera(model, sim.wp_model, sim.wp_data,
                                args.output.parent/'physical-human-camera', 320, 240,
                                camera_id=model.ncam-1)
        np.savez_compressed(args.output.parent/'physical-human-state.npz', qpos=qpos,qvel=qvel)
        return {'xml_path':str(path), 'xml_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'worlds':4, 'nq':model.nq, 'nv':model.nv, 'nu':model.nu,
                'steps':100, 'simulated_seconds':.2, 'free_root_unactuated':True,
                'ball_joint_count':20, 'control_torque_nm':.5,
                'maximum_control_response_velocity':response,
                'maximum_quaternion_norm_error':quaternion_norm_error,
                'root_heights_m':qpos[:,2].tolist(), 'camera':camera,
                'scope':'Finite unassisted dynamics and torque transmission only; no balance or tracking pass.'}

    stage('torch_forward_backward_adam', torch_train)
    # Independent reporting allows Warp to be audited even if the Torch wheel lacks sm_70.
    stage('warp_cuda_kernel', warp_kernel)
    stage('mjlab_batched_contacts_and_camera', batched_contacts)
    if args.scene:
        stage('actual_human_g1_warp_camera', actual_scene)
    if args.physical_model:
        stage('physical_human_ball_motors', physical_human)
    report['status'] = 'passed' if all(s['status']=='passed' for s in report['stages'].values()) else 'failed'
    save()
    return 0 if report['status'] == 'passed' else 1


def _double_kernel(source: 'wp.array(dtype=wp.float32)', destination: 'wp.array(dtype=wp.float32)'):
    index = wp.tid()
    destination[index] = 2.0 * source[index]


def _render_camera(model, wm, wd, prefix, width, height, camera_id=0):
    import mujoco
    import mujoco_warp as mjwarp
    import warp as wp
    from PIL import Image
    active = [False] * model.ncam
    active[camera_id] = True
    context = mjwarp.create_render_context(mjm=model, nworld=wd.nworld, cam_res=[(width,height)],
        cam_active=active, render_rgb=[True], render_depth=[True], render_seg=[True],
        use_textures=True, use_shadows=True, enabled_geom_groups=[0,1,2],
        use_precomputed_rays=True, render_skybox=False, background_color=(.1,.15,.2,1.))
    mjwarp.refit_bvh(wm, wd, context)
    mjwarp.render(wm, wd, context)
    wp.synchronize()
    packed = context.rgb_data.numpy().reshape(wd.nworld, height, width)
    rgb = np.stack([(packed >> shift) & 255 for shift in (16,8,0)], axis=-1).astype(np.uint8)
    depth = context.depth_data.numpy().reshape(wd.nworld,height,width)
    seg = context.seg_data.numpy().reshape(wd.nworld,height,width,2)
    assert np.isfinite(depth).all() and float(depth.max())>0
    assert rgb.max() > rgb.min()
    assert np.any(seg[...,1] == int(mujoco.mjtObj.mjOBJ_GEOM))
    geom_pixels = seg[...,1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    geom_ids = np.unique(seg[...,0][geom_pixels])
    assert len(geom_ids) >= 2
    if any(model.geom(int(i)).name == 'ball_geom' for i in geom_ids):
        ball = (seg[...,0] == model.geom('ball_geom').id) & geom_pixels
        color = rgb[ball].mean(axis=0)
        assert color[0] > color[2] + 30, 'Packed RGB channels must decode R from bits 16..23.'
    Image.fromarray(rgb[0]).save(prefix.with_suffix('.png'))
    np.savez_compressed(prefix.with_suffix('.npz'), rgb=rgb, depth=depth, segmentation=seg)
    return {'rgb_shape': list(rgb.shape), 'depth_shape': list(depth.shape), 'segmentation_shape': list(seg.shape),
            'visible_geom_names': [model.geom(int(i)).name for i in geom_ids],
            'depth_min_m': float(depth.min()), 'depth_max_m': float(depth.max()),
            'rgb_standard_deviation': float(rgb.std()), 'image': str(prefix.with_suffix('.png'))}


if __name__ == '__main__':
    # Warp resolves annotations and global symbols when it compiles this function.
    import warp as wp
    _double_kernel.__annotations__ = {'source': wp.array(dtype=wp.float32),
                                     'destination': wp.array(dtype=wp.float32)}
    raise SystemExit(main())
