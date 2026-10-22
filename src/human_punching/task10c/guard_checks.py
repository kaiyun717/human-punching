"""Brief physical launch/recovery guard holds; diagnostic only, with no learning."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch
import warp as wp

from .commands import FAMILY_ACTIVE_HAND
from .env_cfg import ROOT, make_env_cfg
from .environment import PunchEnv
from .references import StoredReference
from .robot import H2_XML


def render_pair(model, poses, path, family):
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0, 0, 0.95]
    camera.distance, camera.azimuth, camera.elevation = 3.3, 135, -12
    with imageio.get_writer(path, fps=50) as writer:
        for frame, pair in enumerate(poses):
            panels = []
            for endpoint, qpos in zip(('launch', 'recovery'), pair):
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera)
                image = Image.fromarray(renderer.render())
                ImageDraw.Draw(image).text((10, 10), f'{family} {endpoint} guard | constant target | t={frame*.02:.2f}s',
                                           fill='white', stroke_width=1, stroke_fill='black')
                panels.append(np.asarray(image))
            writer.append_data(np.concatenate(panels, axis=1))
    renderer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--families', nargs='+', choices=tuple(FAMILY_ACTIVE_HAND), default=list(FAMILY_ACTIVE_HAND))
    parser.add_argument('--seconds', type=float, default=2.0)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / 'outputs/task10c')
    output.mkdir(parents=True, exist_ok=True)
    if 'SLURM_JOB_ID' not in os.environ or not torch.cuda.is_available():
        raise RuntimeError('Physical guard diagnostics require an allocated CUDA GPU')
    wp.config.kernel_cache_dir = os.environ['WARP_CACHE_PATH']
    result = {'job_id': os.environ['SLURM_JOB_ID'], 'reference_root': str(args.reference_root.resolve()),
              'gpu': torch.cuda.get_device_name(0), 'seed': 102201,
              'selected_model_sha256': hashlib.sha256(H2_XML.read_bytes()).hexdigest(),
              'physics_dt': .005, 'actor_dt': .02, 'constant_pose_control': True,
              'learning_updates': 0, 'diagnostic_not_qualification': True, 'families': {}}
    for family in args.families:
        path = args.reference_root / f'{family}.npz'
        reference = StoredReference.load(path)
        cfg = make_env_cfg(family=family, reference_path=path, num_envs=2, seed=102201, recovery_fraction=0)
        cfg.terminations = {}  # Observe the requested short hold even if posture fails.
        cfg.rewards = {}
        env = PunchEnv(cfg, device='cuda:0')
        env.reset()
        command = env.command_manager.get_term('punch')
        term = env.action_manager.get_term('joint_pos')
        qpos = torch.as_tensor(reference.arrays['qpos'][[0, -1]], dtype=torch.float32, device=env.device)
        origins = env.scene.env_origins.clone()
        qpos[:, :3] += origins
        with env._stream():
            env.sim.data.qpos[:] = qpos
            env.sim.data.qvel.zero_()
            env.sim.forward()
            target = command.robot.data.joint_pos.clone()
            action = (target[:, term.target_ids] - term.offset) / term.scale
            command.previous_glove[:] = command.robot.data.site_pos_w[:, command.site_ids]
            state_parity = float((target - qpos[:, 7:]).abs().max())
            action_parity = float((term.offset + term.scale * action - target[:, term.target_ids]).abs().max())
            assert state_parity < 1e-6 and action_parity < 1e-6, (state_parity, action_parity)
        local = qpos.cpu().numpy().copy()
        local[:, :3] -= origins.cpu().numpy()
        poses, heights, uprights, contacts = [local], [], [], []
        first_fall = [None, None]
        with torch.no_grad():
            for step in range(int(np.ceil(args.seconds / env.step_dt))):
                obs, reward, _, _, _ = env.step(action)
                if not torch.isfinite(obs['actor']).all():
                    raise FloatingPointError(f'Nonfinite {family} guard observation')
                local = env.sim.data.qpos[:].cpu().numpy().copy()
                local[:, :3] -= origins.cpu().numpy()
                poses.append(local)
                height = command.robot.data.root_link_pos_w[:, 2].cpu().numpy().copy()
                upright = (-command.robot.data.projected_gravity_b[:, 2]).cpu().numpy().copy()
                heights.append(height)
                uprights.append(upright)
                n = min(int(env.sim.data.nacon[0]), env.sim.wp_data.naconmax)
                world_ids = wp.to_torch(env.sim.wp_data.contact.worldid)[:n]
                contacts.append([int((world_ids == i).sum()) for i in range(2)])
                for i in range(2):
                    if first_fall[i] is None and (height[i] < .55 or upright[i] < .2):
                        first_fall[i] = (step + 1) * env.step_dt
        poses, heights, uprights, contacts = map(np.asarray, (poses, heights, uprights, contacts))
        np.savez_compressed(output / f'{family}_guards.npz', qpos=poses, root_height=heights,
                            root_upright=uprights, detected_contacts=contacts, actor_dt=.02)
        render_pair(env.sim.mj_model, poses, output / f'{family}_guards.mp4', family)
        result['families'][family] = {'reference_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'initial_joint_state_max_error_rad': state_parity,
            'initial_action_target_max_error_rad': action_parity,
            'action_joint_names': list(term.target_names), 'action_scale': term.scale[0].cpu().tolist(),
            'force_range_actuator_names': [env.sim.mj_model.actuator(i).name for i in range(env.sim.mj_model.nu)],
            'actuator_force_ranges': env.sim.mj_model.actuator_forcerange.tolist(),
            'endpoints': {endpoint: {
            'seconds': len(heights) * env.step_dt, 'first_physical_fall_s': first_fall[i],
            'root_height_initial_m': float(poses[0, i, 2]),
            'root_height_final_m': float(heights[-1, i]), 'root_height_min_m': float(heights[:, i].min()),
            'root_upright_final': float(uprights[-1, i]),
            'root_xy_displacement_m': float(np.linalg.norm(poses[-1, i, :2] - poses[0, i, :2])),
            'detected_contacts_min': int(contacts[:, i].min()), 'detected_contacts_max': int(contacts[:, i].max()),
        } for i, endpoint in enumerate(('launch', 'recovery'))}}
        (output / 'guard_checks.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        env.close()
        del env, command, term, obs, action
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
