"""Load/reset/step every family in allocated CUDA; write one compact JSON file."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import torch
import warp as wp

from .commands import FAMILY_ACTIVE_HAND
from .env_cfg import ROOT, make_env_cfg
from .environment import PunchEnv
from .events import reference_clock
from .mdp import OBSERVATION_SCHEMA
from .robot import JOINT_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='Output JSON path under outputs/task10c')
    parser.add_argument('--families', nargs='+', choices=tuple(FAMILY_ACTIVE_HAND),
                        default=list(FAMILY_ACTIVE_HAND))
    parser.add_argument('--reference-dir', type=Path,
                        help='Explicit selected references; otherwise use configs/task10c/references.json')
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / 'outputs/task10c')
    output.parent.mkdir(parents=True, exist_ok=True)
    if 'SLURM_JOB_ID' not in os.environ or not torch.cuda.is_available():
        raise RuntimeError('Family checks require an allocated CUDA GPU')
    wp.config.kernel_cache_dir = os.environ['WARP_CACHE_PATH']
    results = {'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(0),
               'actor_steps_per_family': 2, 'num_envs': 2, 'families': {}}
    expected_obs = sum(size for _, size in OBSERVATION_SCHEMA)
    for family in args.families:
        index = list(FAMILY_ACTIVE_HAND).index(family)
        hand = FAMILY_ACTIVE_HAND[family]
        reference_path = (args.reference_dir / f'{family}.npz').resolve() if args.reference_dir else None
        cfg = make_env_cfg(family=family, reference_path=reference_path,
                          num_envs=2, seed=102200 + index, recovery_fraction=0.0)
        reference_path = Path(cfg.commands['punch'].reference_path).resolve()
        env = PunchEnv(cfg, device='cuda:0')
        with torch.no_grad():
            obs, _ = env.reset()
            command = env.command_manager.get_term('punch')
            term = env.action_manager.get_term('joint_pos')
            reference = command.reference
            hold_time, hold_rate = reference_clock(
                command.state.requested_ttc + reference.recovery + 0.1,
                command.state.requested_ttc, reference.strike_time, reference.end_time,
                reference.recovery)
            expected_horizon = torch.ceil((command.state.requested_ttc + reference.recovery + .4)
                                          / env.step_dt) * env.step_dt
            checks = {
                'selected_dimensions': (env.sim.mj_model.nq, env.sim.mj_model.nv, env.sim.mj_model.nu) == (38, 37, 31),
                'named_31_actions': tuple(term.target_names) == tuple(JOINT_NAMES) == tuple(command.robot.joint_names),
                'family_and_active_hand': command.reference.metadata['family'] == family
                    and command.reference.metadata['active_hand'] == hand
                    and command.active_index == (0 if hand == 'left' else 1),
                'physics_actor_timing': env.physics_dt == 0.005 and env.step_dt == 0.02,
                'finite_178_actor_critic': expected_obs == 178,
                'finite_rewards': True,
                'terminal_target_is_final_reference': bool(torch.allclose(
                    command.terminal['joint_pos'], reference.arrays['joint_pos'][-1:].expand(2, -1),
                    atol=1e-6, rtol=0)) and bool((command.terminal['joint_vel'] == 0).all()),
                'recovery_duration_matches_reference': abs(reference.recovery
                    - (reference.end_time - reference.strike_time)) < 2e-6,
                'terminal_clock_and_horizon': bool(torch.allclose(hold_time,
                    torch.full_like(hold_time, reference.end_time), atol=1e-6, rtol=0))
                    and bool((hold_rate == 0).all())
                    and bool(torch.allclose(command.horizon, expected_horizon, atol=1e-6, rtol=0)),
            }
            for group in ('actor', 'critic'):
                checks['finite_178_actor_critic'] &= tuple(obs[group].shape) == (2, 178) and bool(torch.isfinite(obs[group]).all())
            action = (command.ref['joint_pos'][:, term.target_ids] - term.offset) / term.scale
            for _ in range(2):
                obs, reward, terminated, truncated, _ = env.step(action)
                checks['finite_rewards'] &= bool(torch.isfinite(reward).all())
                for group in ('actor', 'critic'):
                    checks['finite_178_actor_critic'] &= tuple(obs[group].shape) == (2, 178) and bool(torch.isfinite(obs[group]).all())
            checks['two_actor_steps_eight_physics_steps'] = bool((command.state.physics_steps == 8).all())
            count = min(int(env.sim.data.nacon[0]), env.sim.wp_data.naconmax)
            world_ids = wp.to_torch(env.sim.wp_data.contact.worldid)[:count]
            detected_contacts = [int((world_ids == i).sum()) for i in range(env.num_envs)]
            results['families'][family] = {'active_hand': hand, 'checks': checks,
                'seed': 102200 + index,
                'reference_path': str(reference_path),
                'reference_sha256': hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                'reference_metadata_sha256': hashlib.sha256(reference_path.with_suffix('.json').read_bytes()).hexdigest(),
                'reference_strike_time_s': command.reference.strike_time,
                'reference_end_time_s': reference.end_time,
                'reference_recovery_duration_s': reference.recovery,
                'command_horizons_s': command.horizon.cpu().tolist(),
                'detected_contacts_per_environment_after_two_steps': detected_contacts,
                'terminated_after_two_steps': int(terminated.sum()), 'truncated_after_two_steps': int(truncated.sum())}
        env.close()
        output.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
        assert all(checks.values()), results['families'][family]
        del env, command, term, obs, action
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
