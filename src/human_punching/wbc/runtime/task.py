"""Concrete human WBC environment and unchanged RSL-RL PPO configuration."""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import mdp as mjmdp
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from wbc_mjlab.env.mdp.sampling import RsiCfg
from wbc_mjlab.env.wbc_env_cfg import make_base_wbc_env_cfg
from wbc_mjlab.presets.wbc import apply_wbc
from wbc_mjlab.robots.g1.rl_cfg import g1_wbc_rl_cfg

from . import mdp
from .action import PunchJointPositionActionCfg
from .body import build_human_entity_cfg, build_human_terrain_cfg
from .bundle import load_runtime_bundle
from .command import PunchMotionCommandCfg

TASK_ID = "HumanPunching-WBC-FourFamily-v0"


def human_ppo_cfg():
    cfg = g1_wbc_rl_cfg()
    cfg.experiment_name = "human_punching_wbc"
    cfg.clip_actions = 1.0
    cfg.logger = "tensorboard"
    cfg.upload_model = False
    return cfg


def build_human_env_cfg(bundle, num_envs=1, evaluation=True):
    """Build an inspectable config; never initialize simulation or start learning.

    ``num_envs`` is PER RANK, as required by mjlab's distributed launcher.
    Unqualified data are permitted only for non-learning evaluation fixtures.
    """
    bundle = load_runtime_bundle(bundle)
    if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
        raise ValueError("num_envs must be a positive per-rank integer")
    if not evaluation:
        bundle.require_training_qualification()
    cfg = make_base_wbc_env_cfg(use_reference_residual_action=True)
    cfg.scene.num_envs = num_envs
    cfg.scene.terrain = build_human_terrain_cfg(bundle)
    cfg.scene.entities = {"robot": build_human_entity_cfg(bundle)}
    cfg.viewer.body_name = "Hips"
    cfg.viewer.entity_name = "robot"
    cfg.commands["motion"] = PunchMotionCommandCfg(
        bundle_path=str(bundle.root), motion_file=str(bundle.root / "motions"),
        entity_name="robot", anchor_body_name="Hips", body_names=bundle.body_names,
        actuated_joint_names=bundle.joint_names, resampling_time_range=(1.e9, 1.e9),
        pose_range={}, velocity_range={}, joint_position_range=(0., 0.),
        rsi=RsiCfg(sampling_mode="start"), assistive_wrench_enabled=False,
        ground_reference=False,
    )
    apply_wbc(cfg, motion_body_names=bundle.body_names,
              ee_termination_bodies=("LeftFoot", "RightFoot", "LeftHand", "RightHand"))
    cfg.commands["motion"].assistive_wrench_enabled = False
    cfg.commands["motion"].rsi = replace(cfg.commands["motion"].rsi,
        sampling_mode="start", persist_failure_levels=False, min_bin_span_ratio=0.)
    # Assistance, pushes, reset state editing and encoder/plant randomization
    # require separate qualified curricula; none is inherited accidentally.
    cfg.events = {}
    cfg.actions["joint_pos"] = PunchJointPositionActionCfg(
        # Pinned mjlab resolves this field against actuated JOINT names, despite
        # its name. Native XML actuator names have a separate _native_pd suffix.
        entity_name="robot", actuator_names=bundle.joint_names, scale=.25,
        command_name="motion", preserve_order=True,
    )
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        terms["base_ang_vel"] = ObservationTermCfg(func=mjmdp.base_ang_vel)
        terms["base_lin_vel"] = ObservationTermCfg(func=mjmdp.base_lin_vel)
        terms["actions"] = ObservationTermCfg(func=mjmdp.last_action,
                                               params={"action_name": "joint_pos"})
        terms["punch_condition"] = ObservationTermCfg(func=mdp.punch_condition)
        terms["future_joint_reference"] = ObservationTermCfg(func=mdp.future_joint_reference)
        cfg.observations[group].enable_corruption = False
    cfg.scene.sensors = (
        ContactSensorCfg(name="feet_ground_contact",
            primary=ContactMatch(mode="subtree", pattern="^(LeftFoot|RightFoot)$", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True),
        ContactSensorCfg(name="keybodies_ground_contact",
            primary=ContactMatch(mode="body", pattern="^(" + "|".join(bundle.body_names) + ")$",
                                 entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            # Pattern expansion creates one primary per body. Native netforce
            # reduction has exactly one slot for each primary, not B slots.
            fields=("found", "force"), reduce="netforce", num_slots=1),
    )
    cfg.rewards["foot_slip"].params["asset_cfg"] = SceneEntityCfg(
        "robot", site_names=("LeftFoot_sole", "RightFoot_sole"))
    cfg.rewards["foot_slip"].weight = -.1
    cfg.rewards["timed_target"] = RewardTermCfg(func=mdp.timed_target_reward, weight=2.)
    cfg.terminations["cycle_complete"] = TerminationTermCfg(func=mdp.cycle_finished)
    cfg.sim.mujoco.timestep = .005
    cfg.sim.mujoco.integrator = "implicitfast"
    cfg.sim.mujoco.cone = "pyramidal"
    cfg.sim.mujoco.solver = "newton"
    cfg.sim.mujoco.iterations = 10
    cfg.sim.mujoco.ls_iterations = 20
    cfg.sim.mujoco.tolerance = 1.e-8
    cfg.sim.nconmax = 128
    cfg.sim.njmax = 512
    cfg.decimation = 4
    cfg.episode_length_s = 10.
    cfg.auto_reset = not evaluation
    cfg.is_finite_horizon = True
    return cfg


def register_human_task(bundle, num_envs=8192, evaluation=False, runner_cls=None):
    """Register locally; the caller owns launcher and qualification authorization."""
    from mjlab.tasks.registry import register_mjlab_task
    from wbc_mjlab.extension import WbcRobotSpec, register_robot
    from wbc_mjlab.motion.robot_assets import RobotMotionSpec

    from .runner import HumanMotionTrackingRunner

    checked = load_runtime_bundle(bundle)

    def make_env(*, play=False, task_id=None):
        del task_id
        return build_human_env_cfg(checked, num_envs=num_envs, evaluation=play or evaluation)

    register_robot(WbcRobotSpec(
        robot_id="soma23-human", make_env_cfg=make_env, make_rl_cfg=human_ppo_cfg,
        motion_spec=RobotMotionSpec(scene_cfg_fn=lambda: make_env(play=True).scene,
            actuated_joint_names=checked.joint_names,
            foot_body_names=("LeftFoot", "RightFoot"), mjcf_path=checked.model_path),
        project_root=checked.root,
    ))
    register_mjlab_task(TASK_ID, env_cfg=make_env(), play_env_cfg=make_env(play=True),
                       rl_cfg=human_ppo_cfg(), runner_cls=runner_cls or HumanMotionTrackingRunner)
    return TASK_ID
