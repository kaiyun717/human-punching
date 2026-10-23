"""H2 adaptation of pinned TaskNPoint whole-body tracking, with pure tensor APIs.

Source: TaskNPoint b8faaf284e0cef8021b94e2219492dbd54efba0c,
goal_cond_tracking/mdp/{commands,rewards}.py. Body pose alignment, quaternion
errors, world velocity convention, widths and relative weights are retained.
The intentional changes are height-only anchor position guidance, the H2 body
map (including both head joints), and finite-attempt/recovery budget scaling.
All outputs are soft scores or reward rates, never acceptance conditions.
The optional launch_registered control keeps supplied world poses fixed to the
launch scene instead of applying the usual ongoing actual-pelvis pose alignment.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from mjlab.utils.lab_api.math import (
    quat_apply, quat_error_magnitude, quat_inv, quat_mul, yaw_quat,
)

TASKNPOINT_REVISION = 'b8faaf284e0cef8021b94e2219492dbd54efba0c'
H2_TRACKED_BODY_NAMES = (
    'pelvis',
    'left_hip_roll_link', 'left_knee_link', 'left_ankle_pitch_link',
    'right_hip_roll_link', 'right_knee_link', 'right_ankle_pitch_link',
    'torso_link',
    'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_yaw_link',
    'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_yaw_link',
    'head_pitch_link', 'head_yaw_link',
)
IMITATION_WEIGHTS = {
    'anchor_height': 0.5,
    'anchor_orientation': 0.5,
    'body_position': 1.0,
    'body_orientation': 1.0,
    'body_linear_velocity': 1.0,
    'body_angular_velocity': 1.0,
}
IMITATION_STDS = dict(zip(IMITATION_WEIGHTS, (0.3, 0.4, 0.3, 0.4, 1.0, 3.14)))
COMMAND_EXCESS_SCALE_RAD = 0.3
COMMAND_EXCESS_SMOOTH_L1_BETA = 1.0
COMMAND_JOINT_GROUPS = {
    'head': ('head_pitch_joint', 'head_yaw_joint'),
    'waist': ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'),
    **{f'{side}_leg': tuple(f'{side}_{part}_joint' for part in (
        'hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_roll', 'ankle_pitch'))
       for side in ('left', 'right')},
    **{f'{side}_arm': tuple(f'{side}_{part}_joint' for part in (
        'shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
        'wrist_roll', 'wrist_pitch', 'wrist_yaw'))
       for side in ('left', 'right')},
}


def align_reference_poses(
    actual_pos: torch.Tensor, actual_quat: torch.Tensor,
    reference_pos: torch.Tensor, reference_quat: torch.Tensor,
    anchor_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TaskNPoint's yaw-only pose alignment; all quaternions are wxyz.

    Inputs have shape [batch, bodies, 3/4]. Reference body origins are rotated
    around the reference anchor, placed at actual anchor XY and reference Z.
    Full quaternion orientation is retained after the yaw alignment. Inputs
    are never mutated, and no alignment is applied to velocity references.
    """
    anchor_pos = reference_pos[:, anchor_index, None, :]
    delta_pos = actual_pos[:, anchor_index, None, :].clone()
    delta_pos[..., 2] = anchor_pos[..., 2]
    delta_quat = yaw_quat(quat_mul(
        actual_quat[:, anchor_index], quat_inv(reference_quat[:, anchor_index])))
    delta_quat = delta_quat[:, None, :].expand_as(reference_quat)
    return (
        delta_pos + quat_apply(delta_quat, reference_pos - anchor_pos),
        quat_mul(delta_quat, reference_quat),
    )


def whole_body_errors(
    actual_pos: torch.Tensor, actual_quat: torch.Tensor,
    actual_lin_vel: torch.Tensor, actual_ang_vel: torch.Tensor,
    reference_pos: torch.Tensor, reference_quat: torch.Tensor,
    reference_lin_vel: torch.Tensor, reference_ang_vel: torch.Tensor,
    anchor_index: int = 0,
    *, imitation_frame: str = 'relative',
) -> dict[str, torch.Tensor]:
    """Return squared errors before exponentiation, one tensor per component.

    XYZ squared errors are summed before the mean over tracked bodies, exactly
    as in TaskNPoint. Velocity errors compare body-link world vectors directly;
    they are not rotated by the pose alignment and not root-velocity-relative.
    """
    if imitation_frame == 'relative':
        aligned_pos, aligned_quat = align_reference_poses(
            actual_pos, actual_quat, reference_pos, reference_quat, anchor_index)
    elif imitation_frame == 'launch_registered':
        # The caller supplies the stored trajectory plus its frozen scene
        # translation. Never recenter it on the current learned pelvis.
        aligned_pos, aligned_quat = reference_pos, reference_quat
    else:
        raise ValueError(f'Unknown imitation frame: {imitation_frame!r}')
    return {
        'anchor_height': (reference_pos[:, anchor_index, 2]
                          - actual_pos[:, anchor_index, 2]).square(),
        'anchor_orientation': quat_error_magnitude(
            reference_quat[:, anchor_index], actual_quat[:, anchor_index]).square(),
        'body_position': (aligned_pos - actual_pos).square().sum(-1).mean(-1),
        'body_orientation': quat_error_magnitude(aligned_quat, actual_quat).square().mean(-1),
        'body_linear_velocity': (reference_lin_vel - actual_lin_vel).square().sum(-1).mean(-1),
        'body_angular_velocity': (reference_ang_vel - actual_ang_vel).square().sum(-1).mean(-1),
    }


def whole_body_scores(
    actual_pos: torch.Tensor, actual_quat: torch.Tensor,
    actual_lin_vel: torch.Tensor, actual_ang_vel: torch.Tensor,
    reference_pos: torch.Tensor, reference_quat: torch.Tensor,
    reference_lin_vel: torch.Tensor, reference_ang_vel: torch.Tensor,
    anchor_index: int = 0,
    *, imitation_frame: str = 'relative',
) -> dict[str, torch.Tensor]:
    """Six original exponential kernels, with the declared H2 anchor change."""
    errors = whole_body_errors(
        actual_pos, actual_quat, actual_lin_vel, actual_ang_vel,
        reference_pos, reference_quat, reference_lin_vel, reference_ang_vel, anchor_index,
        imitation_frame=imitation_frame)
    return {name: torch.exp(-error / IMITATION_STDS[name] ** 2)
            for name, error in errors.items()}


def imitation_rates(
    scores: Mapping[str, torch.Tensor], horizon: torch.Tensor,
    guard_weight: torch.Tensor, guard_normalizer: torch.Tensor,
    imitation_budget: float = 7.0, recovery_budget: float = 6.0,
) -> dict[str, torch.Tensor]:
    """Per-component rates; the environment applies dt exactly once.

    The full scheduled actor grid, not elapsed survival or recovery snippets,
    determines H and Z. Perfect scores integrate to budgets 7+6 by default.
    """
    rate = (imitation_budget / horizon
            + recovery_budget * guard_weight / guard_normalizer) / sum(IMITATION_WEIGHTS.values())
    return {name: rate * weight * scores[name] for name, weight in IMITATION_WEIGHTS.items()}


def named_joint_groups(joint_names: Sequence[str]) -> dict[str, tuple[int, ...]]:
    """Map and cover all 31 H2 commands exactly once, including both arms."""
    names = tuple(joint_names)
    expected = tuple(name for group in COMMAND_JOINT_GROUPS.values() for name in group)
    if len(names) != len(expected) or len(set(names)) != len(names) or set(names) != set(expected):
        raise ValueError('Command excess requires every named H2 joint exactly once')
    return {group: tuple(names.index(name) for name in group_names)
            for group, group_names in COMMAND_JOINT_GROUPS.items()}


def command_excess_terms(
    targets: torch.Tensor, hard_limits: torch.Tensor,
    groups: Mapping[str, Sequence[int]], scale_rad: float = COMMAND_EXCESS_SCALE_RAD,
) -> dict[str, torch.Tensor]:
    """Dimensionless SmoothL1 means by group, from physical target excess.

    Hard limits are [joints,2] or [batch,joints,2], in the same named order as
    targets. Beta=1 applies after dividing excess radians by the 0.3-rad scale.
    """
    if scale_rad <= 0:
        raise ValueError('Command excess angular scale must be positive')
    excess = (hard_limits[..., 0] - targets).clamp_min(0)
    excess = excess + (targets - hard_limits[..., 1]).clamp_min(0)
    scaled = excess / scale_rad
    cost = torch.nn.functional.smooth_l1_loss(
        scaled, torch.zeros_like(scaled), reduction='none', beta=COMMAND_EXCESS_SMOOTH_L1_BETA)
    return {name: cost[:, list(ids)].mean(-1) for name, ids in groups.items()}


def command_excess_loss(
    targets: torch.Tensor, hard_limits: torch.Tensor,
    groups: Mapping[str, Sequence[int]], scale_rad: float = COMMAND_EXCESS_SCALE_RAD,
) -> torch.Tensor:
    """Sum six group means (no outer division by six)."""
    return torch.stack(tuple(command_excess_terms(targets, hard_limits, groups, scale_rad).values()), dim=-1).sum(-1)
