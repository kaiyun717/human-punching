"""Focused native event/clock integration and short physical setup observations."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
import warp as wp

from .env_cfg import make_env_cfg, ROOT
from .environment import PunchEnv
from .events import physics_event_callback
from .robot import GLOVE_SITE_NAMES, H2_JOINT_NAMES


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def site_parity(env):
    data = mujoco.MjData(env.sim.mj_model)
    data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
    mujoco.mj_kinematics(env.sim.mj_model, data)
    ids = [env.sim.mj_model.site('robot/' + n).id for n in GLOVE_SITE_NAMES]
    c = env.command_manager.get_term('punch')
    return float(np.max(np.abs(data.site_xpos[ids] - c.robot.data.site_pos_w[0, c.site_ids].cpu().numpy())))


def video(env, qposes, path, title, targets=None):
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = env.scene.env_origins[0].cpu().numpy() + [0, 0, 0.9]
    camera.distance, camera.azimuth, camera.elevation = 3.1, 135, -12
    from PIL import Image, ImageDraw
    with imageio.get_writer(path, fps=50) as writer:
        for i, q in enumerate(qposes):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera)
            if targets is not None:
                # A renderer-only sphere has no compiled collision response.
                geom = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([.06, .06, .06]), np.asarray(targets[i]), np.eye(3).ravel(),
                    np.array([1., .2, .2, .35], dtype=np.float32))
                renderer.scene.ngeom += 1
            frame = Image.fromarray(renderer.render())
            ImageDraw.Draw(frame).text((10, 10), title, fill='white', stroke_width=1, stroke_fill='black')
            writer.append_data(np.asarray(frame))
    renderer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.relative_to(ROOT / 'outputs/task10c')
    out.mkdir(parents=True, exist_ok=True)
    if 'SLURM_JOB_ID' not in os.environ or not torch.cuda.is_available():
        raise RuntimeError('Native checks require an allocated CUDA GPU')
    wp.config.kernel_cache_dir = os.environ['WARP_CACHE_PATH']
    from .train import identity, json_write
    json_write(out / 'identity.json', identity())
    started = time.monotonic()
    cfg = make_env_cfg(num_envs=8, seed=102100, recovery_fraction=0.0)
    cfg.auto_reset = False
    env = PunchEnv(cfg, device='cuda:0')
    env.reset()
    c = env.command_manager.get_term('punch')
    action = env.action_manager.get_term('joint_pos')
    model = env.sim.mj_model
    results = {'job_id': os.environ['SLURM_JOB_ID'], 'physics_dt': env.physics_dt,
               'gpu': torch.cuda.get_device_name(0),
               'actor_dt': env.step_dt, 'nq': model.nq, 'nv': model.nv, 'nu': model.nu,
               'joint_names': list(c.robot.joint_names), 'checks': {}}
    checks = results['checks']
    assert (model.nq, model.nv, model.nu) == (38, 37, 31)
    assert set(c.robot.joint_names) == set(H2_JOINT_NAMES)
    checks['selected_dimensions_and_named_joints'] = True
    checks['correct_5ms_20ms'] = env.physics_dt == .005 and env.step_dt == .02
    results['actuator_force_ranges'] = model.actuator_forcerange.tolist()
    results['actuator_names'] = [model.actuator(i).name for i in range(model.nu)]
    results['actuator_joint_names'] = [model.joint(int(model.actuator_trnid[i,0])).name for i in range(model.nu)]
    results['dof_armature'] = model.dof_armature.tolist()
    results['live_pd_actuators'] = [
        {'joint_names': a._target_names, 'kp': a.stiffness[0].cpu().tolist(),
         'kd': a.damping[0].cpu().tolist(), 'effort_limit': a.force_limit[0].cpu().tolist()}
        for a in c.robot._actuators]
    results['action_scale'] = action.scale[0].cpu().tolist()
    results['default_joint_position'] = c.robot.data.default_joint_pos[0].cpu().tolist()
    results['guard_joint_position'] = c.ref['joint_pos'][0].cpu().tolist()
    results['default_guard_max_delta_rad'] = float((c.ref['joint_pos'][0] - c.robot.data.default_joint_pos[0]).abs().max())
    # Direct qpos fixtures exercise the actual Warp callback, not policy success.
    ids = torch.arange(env.num_envs, device=env.device)
    initial_qpos = env.sim.data.qpos.clone()
    glove = c.robot.data.site_pos_w[:, c.site_ids].clone()
    frozen = glove[:, c.active_index].clone() + torch.tensor([.30, 0, 0], device=env.device)
    c.state.reset(ids, frozen, torch.ones(env.num_envs, device=env.device),
                  torch.zeros(env.num_envs, device=env.device), torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
    with env._stream():
        for substep in range(4):
            env.sim.data.qpos[:] = initial_qpos
            if substep == 0:
                env.sim.data.qpos[:, 0] += .30
            physics_event_callback(env)
    checks['first_of_four_substeps_retained'] = bool(torch.allclose(c.state.first_encounter_elapsed, torch.full_like(c.state.elapsed, .005)))
    checks['target_world_unchanged_by_h2_motion'] = bool(torch.equal(frozen, c.state.target_world))
    results['post_kinematics_cpu_parity_max_m'] = site_parity(env)
    checks['refreshed_geometry_matches_integrated_qpos'] = results['post_kinematics_cpu_parity_max_m'] < 2e-5
    first_bonus = c.state.event_bonus()
    second_bonus = c.state.event_bonus()
    checks['event_pulse_consumed_once'] = bool((first_bonus > 0).all() and (second_bonus == 0).all())
    env.reset()
    with env._stream():
        glove = c.robot.data.site_pos_w[:, c.site_ids].clone()
        c.state.reset(ids, glove[:, c.active_index], torch.ones_like(c.state.elapsed),
                      torch.zeros_like(c.state.elapsed), torch.ones_like(c.state.history_known))
        c.pending_launch[:] = True
        c._update_command()
    checks['native_launch_overlap_ttc_zero'] = bool((c.state.first_encounter_elapsed == 0).all() and c.state.pending.all())
    c.state.requested_ttc.fill_(0.6)
    c.state.first_encounter_elapsed.fill_(0.6)
    c.state.pending.fill_(True)
    reward_before = env.reward_manager._episode_sums['encounter'].clone()
    env.reward_manager.compute(dt=env.step_dt)
    checks['native_reward_manager_one_time_bonus_is_10'] = bool(torch.allclose(
        env.reward_manager._episode_sums['encounter'] - reward_before,
        torch.full_like(reward_before, 10.0)))
    # Resetting env 0 must preserve env 1's encounter, pose, frozen target and clock.
    saved = {k: getattr(c.state, k)[1].clone() for k in ('physics_steps','target_world','encountered','pending','first_encounter_elapsed')}
    pose1 = env.sim.data.qpos[1].clone()
    env.reset(env_ids=torch.tensor([0],device=env.device))
    checks['masked_reset_preserves_other_environment'] = bool(torch.equal(pose1, env.sim.data.qpos[1])) and all(
        torch.equal(v, getattr(c.state,k)[1]) for k,v in saved.items())
    env.reset()
    hold_action = (c.ref['joint_pos'][:, action.target_ids] - action.offset) / action.scale
    env.step(hold_action)
    checks['native_actor_step_four_physics_ticks'] = bool((c.state.physics_steps == 4).all())
    checks['native_post_step_clock_20ms'] = bool(torch.allclose(c.state.elapsed, torch.full_like(c.state.elapsed,.02)))
    results['actual_step_site_parity_max_m'] = site_parity(env)
    checks['actual_step_geometry_parity'] = results['actual_step_site_parity_max_m'] < 2e-5
    c.cfg.recovery_fraction = 1.0
    env.reset()
    checks['native_snippet_age_and_unknown_history'] = bool(
        (~c.state.history_known).all() and (c.state.elapsed > c.state.requested_ttc + .10).all()
        and (c.state.remaining < -.10).all() and torch.isnan(c.state.first_encounter_elapsed).all()
        and (~c.state.pending).all())
    checks['native_snippet_joint_pose_velocity_match_same_phase'] = bool(
        torch.allclose(c.robot.data.joint_pos, c.ref['joint_pos'], atol=1e-6)
        and torch.allclose(c.robot.data.joint_vel, c.ref['joint_vel'], atol=1e-6))
    expected_action = (c.ref['joint_pos'][:, action.target_ids] - action.offset) / action.scale
    checks['native_reset_previous_action_matches_initialized_posture'] = bool(
        torch.allclose(env.action_manager.action, expected_action, atol=1e-6)
        and torch.allclose(env.action_manager.prev_action, expected_action, atol=1e-6))
    c.cfg.recovery_fraction = 0.0
    # Short physics observations, with intact self/ground contacts and actual PD.
    results['physical_holds'] = {}
    for label in ('default_action', 'guard_action'):
        env.reset(seed=102100)
        constant = torch.zeros_like(env.action_manager.action) if label == 'default_action' else (
            c.ref['joint_pos'][:, action.target_ids] - action.offset) / action.scale
        poses, heights, contact_counts = [], [], []
        ended = False
        for k in range(100):
            obs, rew, terminated, truncated, extras = env.step(constant)
            assert torch.isfinite(rew).all() and torch.isfinite(obs['actor']).all()
            poses.append(env.sim.data.qpos[0].cpu().numpy().copy())
            heights.append(float(c.robot.data.root_link_pos_w[0,2]))
            if hasattr(env.sim.data, 'ncon'):
                contact_counts.append(int(env.sim.data.ncon[0]))
            if terminated.any() or truncated.any():
                ended = True
                break
        video(env, poses, out / f'{label}.mp4', f'{label}: physical PD, non-solid target',
              targets=[c.state.target_world[0].cpu().numpy()] * len(poses))
        np.savez_compressed(out / f'{label}.npz', qpos=np.asarray(poses), root_height=np.asarray(heights))
        results['physical_holds'][label] = {'actor_steps':len(poses), 'seconds':len(poses)*env.step_dt,
            'initial_height_m':heights[0], 'minimum_height_m':min(heights), 'final_height_m':heights[-1],
            'terminated':ended, 'contact_count_min': min(contact_counts) if contact_counts else None,
            'contact_count_max': max(contact_counts) if contact_counts else None,
            'physical_falls': int(env.termination_manager.get_term('fallen').sum()),
            'completed_horizons': int(env.termination_manager.get_term('complete').sum())}
    results['elapsed_seconds'] = time.monotonic()-started
    dump(out/'native_checks.json',results)
    env.close()
    assert all(checks.values()), results
    print(json.dumps(results,indent=2))

if __name__ == '__main__':
    main()
