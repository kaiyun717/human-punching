"""Task10C configuration adapted from TaskNPoint's tracking_env_cfg structure."""
from copy import deepcopy
import json
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.nan_guard import NanGuardCfg
from mjlab.viewer import ViewerConfig

from . import mdp
from .commands import PunchCommandCfg
from .events import physics_event_callback
from .imitation import IMITATION_WEIGHTS
from .robot import get_h2_robot_cfg, H2_ACTION_SCALE

ROOT = Path(__file__).resolve().parents[3]
REFERENCE_SELECTION = ROOT / 'configs/task10c/references.json'


def selected_reference_path(family):
    """Resolve the explicit project-owned reference selection for one family."""
    selection = json.loads(REFERENCE_SELECTION.read_text())
    if family not in selection:
        raise ValueError(f'Unknown punch family: {family!r}')
    path = (ROOT / selection[family]).resolve()
    path.relative_to(ROOT)
    return path


def make_env_cfg(family='jab', reference_path=None, target_pool_path=None,
                 num_envs=256, seed=10, recovery_fraction=0.2,
                 target_split='train', ttc_range=None, target_fraction=1.0,
                 easy_fraction=0.5, diagnostics_dir=None,
                 reward_mode='whole_body', imitation_budget=7.0,
                 recovery_budget=6.0, command_excess_weight=1.0,
                 easy_ttc_range=None, clock_mode='legacy', ttc_anchor_time=None,
                 post_entry_duration=None, imitation_frame='relative',
                 task_rewards_enabled=True, fixed_target_snapshot_id=None):
    if reward_mode == 'whole_body':
        reward_names = ('encounter', 'interaction', *IMITATION_WEIGHTS, 'command_excess',
                        'action_rate', 'effort', 'joint_margin')
    elif reward_mode == 'legacy':
        reward_names = ('encounter', 'interaction', 'style', 'balance', 'guard',
                        'action_rate', 'effort', 'joint_margin')
    else:
        raise ValueError(f'Unknown reward mode: {reward_mode!r}')
    if not task_rewards_enabled:
        # Event observation/diagnostics still run at every physics substep. The
        # remaining reward budgets are unchanged, with no removed-weight reuse.
        reward_names = tuple(name for name in reward_names if name not in ('encounter', 'interaction'))
    if reference_path is None:
        reference_path = selected_reference_path(family)
    target_pool_path = target_pool_path or ROOT / 'outputs/task10c/setup/target_pool.npz'
    group = ObservationGroupCfg(terms={'task': ObservationTermCfg(func=mdp.observations)},
                                concatenate_terms=True, enable_corruption=False)
    return ManagerBasedRlEnvCfg(
        seed=seed, decimation=4, episode_length_s=20, is_finite_horizon=True,
        scene=SceneCfg(num_envs=num_envs, env_spacing=4.0,
                       terrain=TerrainEntityCfg(terrain_type='plane'),
                       entities={'robot': get_h2_robot_cfg()}),
        sim=SimulationCfg(nconmax=128, njmax=600, mujoco=MujocoCfg(timestep=0.005),
                           nan_guard=NanGuardCfg(enabled=True, buffer_size=2,
                             output_dir=str(diagnostics_dir or ROOT / 'outputs/task10c/setup/numerical_diagnostics'))),
        viewer=ViewerConfig(distance=3.5, elevation=-10, azimuth=135),
        observations={'actor': group, 'critic': deepcopy(group)},
        actions={'joint_pos': JointPositionActionCfg(entity_name='robot', actuator_names=('.*',),
                    scale=H2_ACTION_SCALE, use_default_offset=True)},
        commands={'punch': PunchCommandCfg(reference_path=str(reference_path),
                    target_pool_path=str(target_pool_path), family=family,
                    recovery_fraction=recovery_fraction, target_split=target_split, ttc_range=ttc_range,
                    target_fraction=target_fraction, easy_fraction=easy_fraction,
                    easy_ttc_range=easy_ttc_range, clock_mode=clock_mode,
                    ttc_anchor_time=ttc_anchor_time, post_entry_duration=post_entry_duration,
                    reward_mode=reward_mode, imitation_budget=imitation_budget,
                    recovery_budget=recovery_budget, command_excess_weight=command_excess_weight,
                    imitation_frame=imitation_frame, task_rewards_enabled=task_rewards_enabled,
                    fixed_target_snapshot_id=fixed_target_snapshot_id)},
        events={},
        rewards={name: RewardTermCfg(func=getattr(mdp, name), weight=1.0)
                 for name in reward_names},
        terminations={'complete': TerminationTermCfg(func=mdp.complete, time_out=False),
                       'fallen': TerminationTermCfg(func=mdp.fallen, time_out=False)},
        metrics={'encounter_refresh': MetricsTermCfg(func=physics_event_callback, per_substep=True, reduce='last')},
    )
