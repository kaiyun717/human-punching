"""Thin export adapter: all optimization remains WBC/mjlab/RSL-RL PPO."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.rl.runner import MjlabOnPolicyRunner
from wbc_mjlab.rl.runner import PolicyOnlyMotionTrackingRunner, _tracking_policy_onnx_metadata

from .bundle import load_runtime_bundle, sha256


def require_float32_observations(env):
    """Check cached native inputs without changing history or silently casting."""
    manager = env.observation_manager
    observations = manager.compute(update_history=False)
    audit = {}
    for group in ("actor", "critic"):
        value = observations[group]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Human {group} observations must be a concatenated tensor")
        audit[group] = {"dtype": str(value.dtype), "shape": list(value.shape)}
        if value.dtype != torch.float32:
            terms = {}
            for name in manager.active_terms[group]:
                cfg = manager.get_term_cfg(group, name)
                terms[name] = str(cfg.func(env, **cfg.params).dtype)
            raise TypeError(f"Human {group} observations require torch.float32; "
                            f"received {value.dtype}; term dtypes: {terms}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Human {group} observations contain nonfinite values before learning")
    return audit


class HumanMotionTrackingRunner(PolicyOnlyMotionTrackingRunner):
    def __init__(self, env, *args, **kwargs):
        self.human_observation_dtype_audit = require_float32_observations(env.unwrapped)
        super().__init__(env, *args, **kwargs)

    def learn(self, *args, **kwargs):
        command = self.env.unwrapped.command_manager.get_term("motion")
        load_runtime_bundle(command.cfg.bundle_path).require_training_qualification()
        return super().learn(*args, **kwargs)

    def build_policy_export_metadata(self, run_path):
        env = self.env.unwrapped
        action = env.action_manager.get_term("joint_pos")
        command = env.command_manager.get_term("motion")
        bundle = load_runtime_bundle(command.cfg.bundle_path)
        metadata = _tracking_policy_onnx_metadata(env, action, run_path,
                                                 action_mode="human_reference_residual_200hz")
        metadata.update({
            "anchor_body_name": "Hips", "body_names": list(bundle.body_names),
            "physics_dt": .005, "policy_dt": .020, "reference_dt": .005,
            "raw_action_clip": 1., "reference_bundle_sha256": sha256(bundle.root / "manifest.json"),
            "model_sha256": sha256(bundle.model_path),
            "controller": "native_affine_pd_force_limited",
            "joint_target_clamp": "physical_joint_ranges",
            "previous_action": "clipped_raw_residual",
            "target_commitment": "frozen_world_target_and_plan_at_launch",
        })
        return metadata

    def save(self, path, infos=None):
        # Avoid WBC's warning-only export path: a missing deploy artifact is a
        # failed save operation, while the checkpoint remains available to resume.
        if int(os.environ.get("RANK", "0")) != 0:
            return
        command = self.env.unwrapped.command_manager.get_term("motion")
        bundle = load_runtime_bundle(command.cfg.bundle_path)
        checkpoint_info = {
            **(infos or {}),
            "human_checkpoint_contract": {
                "schema": "human_wbc_checkpoint_v1",
                "reference_manifest_sha256": sha256(bundle.root / "manifest.json"),
                "controller_sha256": bundle.controller_sha256(),
                "resume_qualification": "not_qualified",
                "distributed_rng_scope": "rank_zero_only",
                "physical_state_saved": False,
                "sampler": "uniform_family_then_uniform_variant_guard_start",
                "trajectory_ids": command.trajectory_ids.detach().cpu().clone(),
                "physics_steps": command.physics_steps.detach().cpu().clone(),
                "rng_python": random.getstate(),
                "rng_numpy": np.random.get_state(),
                "rng_torch_cpu": torch.get_rng_state(),
                # A worker sees every allocated device. Reading all CUDA RNGs
                # would initialize peer-device contexts and distort their VRAM.
                "rng_torch_cuda": (torch.cuda.get_rng_state(self.device)
                                   if torch.cuda.is_initialized() else None),
            },
        }
        MjlabOnPolicyRunner.save(self, path, checkpoint_info)
        directory = Path(path).parent / "params"
        self.export_policy_to_onnx(str(directory), "policy.onnx")
        # Accepted robust_sav already includes Torch; this portable actor also
        # embeds normalization and does not require WBC or ONNX Runtime imports.
        self.export_policy_to_jit(str(directory), "policy_jit.pt")
        metadata = self.build_policy_export_metadata(str(Path(path).parent))
        attach_metadata_to_onnx(str(directory / "policy.onnx"), metadata)
        controller = {"schema": "human_wbc_controller_v1", **bundle.controller_contract(),
                      "controller_sha256": bundle.controller_sha256(), "policy_metadata": metadata}
        (directory / "human_controller.json").write_text(json.dumps(controller, indent=2) + "\n")
