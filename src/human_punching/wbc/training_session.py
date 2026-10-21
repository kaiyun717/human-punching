"""Completed-iteration continuation around the inherited WBC learning loop.

The wrapper changes launch/checkpoint bookkeeping only. PPO, observation
normalization, rollout storage and optimization remain in WBC/RSL-RL. Every
session starts physical episodes at guard and ends with a collective full learner
checkpoint. A resumed session starts fresh trajectories, never a saved pose.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from .budget import STAGE_CEILINGS
from .checkpointing import (
    BoundaryProgress, StageTransition, build_checkpoint_contract,
    load_boundary_checkpoint, save_boundary_checkpoint,
)
from .provenance import PROJECT_ROOT, write_json
from .measurement_contract import configuration_value
from .validation import require_bounded_smoke_without_validation

SAMPLER_IDENTITY = {
    "family": "uniform_four_families", "variant": "uniform_within_family",
    "initialization": "guard_start", "reference_state_initialization": False,
    "assistance": False, "live_g1_conditioning": False,
}


def observation_schema(env) -> dict:
    manager = env.observation_manager
    groups = {}
    for group in ("actor", "critic"):
        terms = []
        for name, dimension in zip(manager.active_terms[group],
                                   manager.group_obs_term_dim[group], strict=True):
            cfg = manager.get_term_cfg(group, name)
            terms.append({"name": name, "shape": list(dimension),
                          "function": cfg.func.__module__ + "." + cfg.func.__qualname__,
                          "processing": configuration_value(cfg)})
        groups[group] = {"terms": terms, "shape": list(manager.group_obs_dim[group]),
                         "concatenated": manager.group_obs_concatenate[group],
                         "group_processing": configuration_value(env.cfg.observations[group]),
                         "normalization": "RSL_actor_or_critic_running_normalizer_in_state_dict"}
    return {"schema": "human_wbc_observations_v1", "groups": groups,
            "previous_action": "executed_clipped_normalized_residual",
            "frozen_g1_after_launch": True}


def completed_progress(prior: dict | None, stage: str, iterations: int) -> BoundaryProgress:
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("A session must complete a positive integer iteration count")
    prior = prior or {"completed_iterations": 0, "stage": stage,
                      "stage_completed_iterations": 0, "completed_validation_panels": 0}
    if prior["stage"] != stage:
        raise ValueError("Stage transition must be explicitly restored before learning")
    stage_count = prior["stage_completed_iterations"] + iterations
    if stage_count > STAGE_CEILINGS[stage]:
        raise ValueError("Cumulative session counts exceed the accepted stage ceiling")
    return BoundaryProgress(
        completed_iterations=prior["completed_iterations"] + iterations,
        stage=stage, stage_completed_iterations=stage_count,
        completed_validation_panels=prior.get("completed_validation_panels", 0),
        curriculum=prior.get("curriculum", {}), validation=prior.get("validation", {}))


def make_session_runner(plan: dict, output: Path, learner_configuration: dict):
    """Return a local runner subclass for mjlab's supported task registration.

    The factory runs inside each worker. Its immutable configuration is kept out
    of upstream source and no global method or private registry is patched.
    """
    from .runtime.bundle import load_runtime_bundle
    from .runtime.runner import HumanMotionTrackingRunner

    settings, learner = deepcopy(plan), deepcopy(learner_configuration)

    class HumanSessionRunner(HumanMotionTrackingRunner):
        def learn(self, num_learning_iterations, init_at_random_ep_len=False):
            # mjlab's stock launcher requests randomized timeout counters. That
            # can truncate complete guard-start cycles; the human task declines it.
            del init_at_random_ep_len
            # No native trained-actor validation callback exists yet. Do not let
            # a full production budget silently bypass validation/stopping.
            require_bounded_smoke_without_validation(settings, num_learning_iterations)
            if num_learning_iterations != settings["iterations"]:
                raise ValueError("Inherited launcher iteration count differs from session plan")
            env = self.env.unwrapped
            command = env.command_manager.get_term("motion")
            bundle = load_runtime_bundle(command.cfg.bundle_path)
            bundle.require_training_qualification()
            action = env.action_manager.get_term("joint_pos")
            contract = build_checkpoint_contract(
                body_path=bundle.model_path, controller=bundle.controller_contract(),
                actor_observation_schema=observation_schema(env),
                action_schema={"joint_names": list(action.target_names), "dimension": action.action_dim,
                               "raw_clip": 1., "scale_rad": .25, "clock_s": .020},
                simulation_contract={"mujoco": asdict(env.cfg.sim.mujoco),
                                     "decimation": env.cfg.decimation,
                                     "nconmax": env.cfg.sim.nconmax, "njmax": env.cfg.sim.njmax},
                learner_configuration=learner,
                reference_manifest=bundle.root / "manifest.json",
                implementation_sources={str(p.relative_to(PROJECT_ROOT)): p
                    for p in (PROJECT_ROOT / "src/human_punching").rglob("*.py")})
            prior, restoration = None, None
            if settings.get("resume"):
                resume = settings["resume"]
                transition = settings.get("stage_transition")
                restoration = load_boundary_checkpoint(
                    Path(resume["path"]), self, manifest_sha256=resume["manifest_sha256"],
                    expected_contract=contract, stage=settings["stage"],
                    sampler_identity=SAMPLER_IDENTITY, reset_environment=self.env.reset,
                    transition=StageTransition(**transition) if transition else None)
                prior = restoration["progress"]
            elif settings["stage"] != "canonical":
                raise ValueError("Later training stages must restore a full learner checkpoint")
            progress = completed_progress(prior, settings["stage"], num_learning_iterations)
            if self.gpu_global_rank == 0:
                write_json(output / "session_contract.json", {
                    "contract": asdict(contract), "sha256": contract.sha256,
                    "planned_completed_progress": asdict(progress), "restoration": restoration,
                    "randomized_initial_episode_lengths": False,
                    "physical_state_restored": False,
                    "validation_performed_by_training_loop": False})
            result = super().learn(num_learning_iterations=num_learning_iterations,
                                   init_at_random_ep_len=False)
            writer = self.logger.writer
            if writer is not None:
                if hasattr(writer, "flush"):
                    writer.flush()
                if hasattr(writer, "close"):
                    writer.close()
            receipt = save_boundary_checkpoint(
                output / "boundary_checkpoint", self, contract=contract, progress=progress,
                sampler_identity=SAMPLER_IDENTITY)
            if self.gpu_global_rank == 0:
                write_json(output / "session_complete.json", {
                    "schema": "human_wbc_completed_training_session_v1",
                    "completed_progress": asdict(progress), "checkpoint": receipt,
                    "validation_performed": False, "stage_accepted": False})
            return result

    return HumanSessionRunner
