"""WBC motion command with immutable world references and uninterrupted cycles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from wbc_mjlab.env.mdp.commands import MotionCommand, MotionCommandCfg

from .bundle import FAMILIES, load_runtime_bundle


@dataclass(kw_only=True)
class PunchMotionCommandCfg(MotionCommandCfg):
    bundle_path: str
    final_guard_hold_s: float = 1.0

    def build(self, env):
        return PunchMotionCommand(self, env)


class PunchMotionCommand(MotionCommand):
    """Reuse WBC's library/observations/metrics without clip-wrap state writes.

    The action term owns the 200 Hz integer clock. Command updates synchronize
    WBC's 50 Hz frame indices but never advance that clock or replace a pose.
    References and launch context are fixed for the complete episode.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        bundle = load_runtime_bundle(cfg.bundle_path)
        if tuple(self.robot.joint_names) != bundle.joint_names:
            raise ValueError("Runtime human joint ordering differs from converted references")
        by_name = {bundle.path(c["motion_path"]).stem: c for c in bundle.clips}
        self.clips = tuple(by_name[name] for name in self.motion.segment_names)
        fields = ("qpos", "qvel", "joint_feedforward", "body_pos_w", "body_quat_w",
                  "body_lin_vel_w", "body_ang_vel_w")
        arrays = {name: [] for name in fields}
        arrays.update(anchor_lin_acc_w=[], anchor_ang_acc_w=[])
        starts, lengths, total = [], [], 0
        for clip in self.clips:
            with np.load(bundle.path(clip["master_path"]), allow_pickle=False) as master:
                count = len(master["time_s"])
                starts.append(total)
                lengths.append(count)
                total += count
                for name in fields:
                    value = master[name]
                    if name.startswith("body_"):
                        value = value[:, self.body_indexes.cpu().numpy()]
                    arrays[name].append(value.copy())
                for source, destination in (("body_lin_vel_w", "anchor_lin_acc_w"),
                                            ("body_ang_vel_w", "anchor_ang_acc_w")):
                    anchor_velocity = arrays[source][-1][:, self.motion_anchor_body_index]
                    arrays[destination].append(np.gradient(anchor_velocity, .005, axis=0))
        self.master = {name: torch.as_tensor(np.concatenate(parts), dtype=torch.float32,
                                            device=self.device)
                       for name, parts in arrays.items()}
        self.master_starts = torch.tensor(starts, dtype=torch.long, device=self.device)
        self.master_lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)
        self.physics_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.family_ids = torch.tensor([FAMILIES.index(c["family"]) for c in self.clips],
                                       dtype=torch.long, device=self.device)
        self.targets = torch.tensor([c["target_w"] for c in self.clips],
                                    dtype=torch.float32, device=self.device)
        self.launch_times = torch.tensor([c["launch_time_s"] for c in self.clips],
                                         dtype=torch.float32, device=self.device)
        self.ttcs = torch.tensor([
            c.get("requested_ttc_s") if c.get("requested_ttc_s") is not None
            else c["reference_contact_delay_s"] for c in self.clips],
            dtype=torch.float32, device=self.device)
        self.g1_context = torch.tensor([c.get("launch_features") or [] for c in self.clips],
                                        dtype=torch.float32, device=self.device)
        self._family_clips = [torch.where(self.family_ids == family)[0] for family in range(4)]
        # Repair the inherited loader's concatenation-boundary acceleration
        # derivatives: each reference clip has independent temporal boundaries.
        for start, length in zip(self.motion.segment_start_idx.tolist(),
                                 self.motion.segment_length.tolist(), strict=True):
            span = slice(start, start + length)
            for source, destination in ((self.motion.body_lin_vel_w, self.motion.anchor_lin_acc_w),
                                        (self.motion.body_ang_vel_w, self.motion.anchor_ang_acc_w)):
                destination[span] = torch.gradient(source[span, self.motion_anchor_body_index],
                                                   spacing=env.step_dt, dim=0)[0]

    def sample_master(self, name: str, offset_steps: int = 0) -> torch.Tensor:
        local = (self.physics_steps + offset_steps).clamp(min=0)
        local = torch.minimum(local, self.master_lengths[self.trajectory_ids] - 1)
        value = self.master[name][self.master_starts[self.trajectory_ids] + local]
        if name in ("qvel", "body_lin_vel_w", "body_ang_vel_w",
                    "anchor_lin_acc_w", "anchor_ang_acc_w"):
            finished = self.physics_steps + offset_steps >= self.master_lengths[self.trajectory_ids] - 1
            value = torch.where(finished.reshape(-1, *([1] * (value.ndim - 1))),
                                torch.zeros_like(value), value)
        return value

    @property
    def elapsed(self):
        return self.physics_steps.to(torch.float32) * .005

    @property
    def target_w(self):
        return self.targets[self.trajectory_ids] + self._env.scene.env_origins

    @property
    def joint_pos(self):
        if not hasattr(self, "master"):
            return super().joint_pos
        return self.sample_master("qpos")[:, 7:]

    @property
    def joint_vel(self):
        if not hasattr(self, "master"):
            return super().joint_vel
        return self.sample_master("qvel")[:, 6:]

    @property
    def body_pos_w(self):
        if not hasattr(self, "master"):
            return super().body_pos_w
        return self.sample_master("body_pos_w") + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self):
        if not hasattr(self, "master"):
            return super().body_quat_w
        return self.sample_master("body_quat_w")

    @property
    def body_lin_vel_w(self):
        if not hasattr(self, "master"):
            return super().body_lin_vel_w
        return self.sample_master("body_lin_vel_w")

    @property
    def body_ang_vel_w(self):
        if not hasattr(self, "master"):
            return super().body_ang_vel_w
        return self.sample_master("body_ang_vel_w")

    # The parent anchor properties index its 50 Hz loader directly. Override
    # them so all tracking observations agree with the physical reference clock.
    @property
    def anchor_pos_w(self):
        return self.body_pos_w[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self):
        return self.body_quat_w[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self):
        return self.body_lin_vel_w[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self):
        return self.body_ang_vel_w[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_acc_w(self):
        return self.sample_master("anchor_lin_acc_w")

    @property
    def anchor_ang_acc_w(self):
        return self.sample_master("anchor_ang_acc_w")

    def update_relative_body_poses(self):
        # Deliberately no current-human anchor/yaw re-alignment: the complete
        # plan remains in its committed world placement for strike and recovery.
        self.body_pos_relative_w[:] = self.body_pos_w
        self.body_quat_relative_w[:] = self.body_quat_w

    def _resample_command(self, env_ids):
        # Only CommandManager.reset may initialize physical state. A timer
        # expiry or a user resample inside an episode is a hard error.
        if not getattr(self, "_reset_in_progress", False):
            raise RuntimeError("Punch references may only resample at an episode reset")
        families = torch.randint(4, (len(env_ids),), device=self.device)
        for family, candidates in enumerate(self._family_clips):
            chosen = env_ids[families == family]
            indices = torch.randint(len(candidates), (len(chosen),), device=self.device)
            self.trajectory_ids[chosen] = candidates[indices]
        self.physics_steps[env_ids] = 0
        self.time_steps[env_ids] = self.motion.segment_start_idx[self.trajectory_ids[env_ids]]
        self.episode_assist_gain[env_ids] = 0.
        self._episode_start_step[env_ids] = self.time_steps[env_ids]
        self._episode_start_bin[env_ids] = 0
        self._episode_similarity_sum[env_ids] = 0.
        self._episode_step_count[env_ids] = 0
        self._write_reference_state_to_sim(
            env_ids, self.body_pos_w[env_ids, 0], self.body_quat_w[env_ids, 0],
            self.body_lin_vel_w[env_ids, 0], self.body_ang_vel_w[env_ids, 0],
            self.joint_pos[env_ids], self.joint_vel[env_ids],
        )
        self.update_relative_body_poses()

    def reset(self, env_ids):
        self._reset_in_progress = True
        try:
            return super().reset(env_ids)
        finally:
            self._reset_in_progress = False

    def _write_reference_state_to_sim(self, *args, **kwargs):
        if not getattr(self, "_reset_in_progress", False):
            raise RuntimeError("Physical reference initialization is restricted to episode reset")
        return super()._write_reference_state_to_sim(*args, **kwargs)

    def reset_to_frame(self, env_ids, frame):
        raise RuntimeError("Mid-cycle pose replacement is disabled for the human task")

    def create_gui(self, *args, **kwargs):
        # Upstream's scrubber calls reset_to_frame and can mutate phase. The
        # human evaluator intentionally exposes no reference-state scrubber.
        return None

    def _update_command(self, env_ids):
        ids = slice(None) if env_ids is None else env_ids
        frame = self.physics_steps[ids] // 4
        clip = self.trajectory_ids[ids]
        frame = torch.minimum(frame, self.motion.segment_length[clip] - 1)
        self.time_steps[ids] = self.motion.segment_start_idx[clip] + frame
        self.update_relative_body_poses()
