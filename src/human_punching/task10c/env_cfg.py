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
                 target_split='train', ttc_range=None):
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
                             output_dir=str(ROOT / 'outputs/task10c/setup/numerical_diagnostics'))),
        viewer=ViewerConfig(distance=3.5, elevation=-10, azimuth=135),
        observations={'actor': group, 'critic': deepcopy(group)},
        actions={'joint_pos': JointPositionActionCfg(entity_name='robot', actuator_names=('.*',),
                    scale=H2_ACTION_SCALE, use_default_offset=True)},
        commands={'punch': PunchCommandCfg(reference_path=str(reference_path),
                    target_pool_path=str(target_pool_path), family=family,
                    recovery_fraction=recovery_fraction, target_split=target_split, ttc_range=ttc_range)},
        events={},
        rewards={name: RewardTermCfg(func=getattr(mdp, name), weight=1.0)
                 for name in ('encounter', 'interaction', 'style', 'balance', 'guard',
                              'action_rate', 'effort', 'joint_margin')},
        terminations={'complete': TerminationTermCfg(func=mdp.complete, time_out=False),
                       'fallen': TerminationTermCfg(func=mdp.fallen, time_out=False)},
        metrics={'encounter_refresh': MetricsTermCfg(func=physics_event_callback, per_substep=True, reduce='last')},
    )
